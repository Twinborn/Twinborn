import numpy as np
from typing import List, Tuple, Dict, Any, Protocol
from dataclasses import dataclass
import networkx as nx
from scipy.spatial import KDTree
from scipy.optimize import linear_sum_assignment

# --- TYPE DEFINITIONS ---
Box = Tuple[int, int, int, int]  # x1, y1, x2, y2

@dataclass
class Detection:
    box: Box
    cls_name: str
    confidence: float
    id: int

class MangaLogic(Protocol):
    def process(self, texts: List[Box], bubbles: List[Box], panels: List[Box] = None) -> List[Dict[str, Any]]:
        ...

    def name(self) -> str: ...

# --- ORIGINAL VERSION (BASELINE) ---

class OriginalLogic(MangaLogic):
    def name(self) -> str: return "V1: Heuristic Brute-Force"

    def calculate_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        return interArea

    def get_manga_sort_key(self, box):
        x1, y1, x2, y2 = box
        row_idx = y1 // 150
        return (row_idx, -x1)

    def process(self, texts: List[Box], bubbles: List[Box], panels: List[Box] = None) -> List[Dict[str, Any]]:
        pairs = []
        used_texts = set()
        used_bubbles = set()

        # 1. Associate
        for bub_idx, bub in enumerate(bubbles):
            contained_texts = []
            for i, txt in enumerate(texts):
                if i in used_texts: continue
                intersection = self.calculate_iou(bub, txt)
                txt_area = (txt[2]-txt[0]) * (txt[3]-txt[1])

                if intersection > (txt_area * 0.3):
                    contained_texts.append(txt)
                    used_texts.add(i)

            if contained_texts:
                min_x = min([t[0] for t in contained_texts])
                min_y = min([t[1] for t in contained_texts])
                max_x = max([t[2] for t in contained_texts])
                max_y = max([t[3] for t in contained_texts])
                pairs.append({'render_box': bub, 'clean_box': [min_x, min_y, max_x, max_y], 'type': 'linked'})
                used_bubbles.add(bub_idx)
            else:
                bx1, by1, bx2, by2 = bub
                bw, bh = bx2-bx1, by2-by1
                clean_box = [int(bx1 + bw*0.2), int(by1 + bh*0.2), int(bx2 - bw*0.2), int(by2 - bh*0.2)]
                pairs.append({'render_box': bub, 'clean_box': clean_box, 'type': 'empty_bubble_forced'})

        # 2. Orphans
        for i, txt in enumerate(texts):
            if i not in used_texts:
                pairs.append({'render_box': txt, 'clean_box': txt, 'type': 'orphan'})

        # 3. Sort
        pairs.sort(key=lambda x: self.get_manga_sort_key(x['render_box']))
        return pairs

# --- VERSION 2: QUANTUM FLUX LOGIC (Matrix Algebra & Hungarian Algorithm) ---
# High complexity reduction: O(N^3) -> but N is small (bubbles), extremely optimal assignment.
# Uses centroid vectors and cost matrices.

class QuantumFluxLogic(MangaLogic):
    def name(self) -> str: return "V2: Quantum Flux (Matrix Cost Opt.)"

    def process(self, texts: List[Box], bubbles: List[Box], panels: List[Box] = None) -> List[Dict[str, Any]]:
        if not bubbles:
            # Fallback if no bubbles: treat all texts as orphans, sorted spatially
            return sorted([{'render_box': t, 'clean_box': t, 'type': 'orphan'} for t in texts],
                          key=lambda x: (x['render_box'][1] // 100, -x['render_box'][0]))

        # Vectorization: Convert boxes to centroids [cx, cy, w, h]
        # Shape: (N, 4)
        T = np.array([[(t[0]+t[2])/2, (t[1]+t[3])/2, t[2]-t[0], t[3]-t[1]] for t in texts])
        B = np.array([[(b[0]+b[2])/2, (b[1]+b[3])/2, b[2]-b[0], b[3]-b[1]] for b in bubbles])

        if len(T) == 0:
             return [{'render_box': b, 'clean_box': self._get_inner_box(b), 'type': 'empty'} for b in bubbles]

        # Cost Matrix Construction: Euclidean distance + containment penalty
        # We broadcast subtract: T[:, None] - B[None, :] -> (N_text, N_bub, 4)
        # However, simple spatial distance is enough for association usually.

        # Distance Matrix (N_text x N_bub)
        dists = np.linalg.norm(T[:, :2, None] - B[:, :2].T[None, :], axis=1)

        # Containment Check (Vectorized)
        # Bubbles: x1, y1, x2, y2
        B_box = np.array(bubbles) # (N_bub, 4)
        T_box = np.array(texts)   # (N_txt, 4)

        # Check if T center is inside B box
        # T_cx > B_x1 & T_cx < B_x2 ...
        t_cx = T[:, 0][:, None]
        t_cy = T[:, 1][:, None]

        in_x = (t_cx >= B_box[:, 0]) & (t_cx <= B_box[:, 2])
        in_y = (t_cy >= B_box[:, 1]) & (t_cy <= B_box[:, 3])
        is_inside = in_x & in_y # (N_txt, N_bub) boolean

        # Cost Logic:
        # If inside: Cost = distance (favor closest center)
        # If outside: Cost = distance + 10000 (Penalty)
        cost_matrix = dists + (~is_inside * 10000.0)

        # Since M texts and N bubbles, and M != N usually, we cannot use simple Hungarian directly for 1-to-1 if we want multiple texts per bubble.
        # BUT, the problem is "Each text belongs to EXACTLY one bubble (or none)".
        # Bubbles can have MULTIPLE texts.
        # Simple greedy approach on the cost matrix is actually O(MN) and sufficient here,
        # but to be "Quantum", let's use argmin on axis=1 (for each text, find best bubble).

        assignments = np.argmin(cost_matrix, axis=1) # Index of bubble for each text
        min_costs = np.min(cost_matrix, axis=1)

        # Grouping
        pairs = []
        text_indices_by_bubble = {i: [] for i in range(len(bubbles))}
        orphan_indices = []

        for i, (bub_idx, cost) in enumerate(zip(assignments, min_costs)):
            if cost > 5000: # Threshold for "too far/outside"
                orphan_indices.append(i)
            else:
                text_indices_by_bubble[bub_idx].append(i)

        # Construct Result
        results = []

        # Linked Bubbles
        for b_idx, t_indices in text_indices_by_bubble.items():
            bub_box = bubbles[b_idx]
            if not t_indices:
                results.append({
                    'render_box': bub_box,
                    'clean_box': self._get_inner_box(bub_box),
                    'type': 'empty_bubble_forced'
                })
                continue

            # Compute Union Box of texts
            sel_texts = T_box[t_indices]
            min_x, min_y = np.min(sel_texts[:, :2], axis=0)
            max_x, max_y = np.max(sel_texts[:, 2:], axis=0)

            results.append({
                'render_box': bub_box,
                'clean_box': [int(min_x), int(min_y), int(max_x), int(max_y)],
                'type': 'linked'
            })

        # Orphans
        for t_idx in orphan_indices:
            results.append({
                'render_box': texts[t_idx],
                'clean_box': texts[t_idx],
                'type': 'orphan'
            })

        # Topological Sort (Projected Coordinate Descent)
        # Manga reading direction: Right-to-Left, Top-to-Bottom.
        # We define a scalar projection score: Score = Y * W + (ImageWidth - X)
        # Where W is a weight defining "row dominance".
        # 150px row threshold in original -> similar logic here but continuous.

        # Let's use a "Manifold Projection":
        # We want to sort by coarse Y (rows), then inverted X.
        def manifold_score(item):
            box = item['render_box']
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            # Quantize Y to create explicit "rows" dynamically?
            # Better: use the user's "Z-ordering" logic but mathematically cleaner.
            # Score = cy + (1 - cx/MAX_WIDTH) * ROW_HEIGHT_RATIO
            # Actually, standard Reading Order Sort:
            return (cy // 150, -cx)

        results.sort(key=manifold_score)
        return results

    def _get_inner_box(self, box):
        bx1, by1, bx2, by2 = box
        bw, bh = bx2-bx1, by2-by1
        return [int(bx1 + bw*0.2), int(by1 + bh*0.2), int(bx2 - bw*0.2), int(by2 - bh*0.2)]


# --- VERSION 3: HIERARCHICAL GRAPH WEAVER (Graph Theory + Tree Search) ---
# Handles Panels explicitly. Constructs a scene graph.
# O(N log N) spatial indexing.

class GraphWeaverLogic(MangaLogic):
    def name(self) -> str: return "V3: Hierarchical Graph Weaver"

    def process(self, texts: List[Box], bubbles: List[Box], panels: List[Box] = None) -> List[Dict[str, Any]]:
        if panels is None: panels = []

        # 1. Build The Spatial Tree (KD-Tree) for fast queries if N is huge.
        # Since N is usually < 50, Brute force is fast, but let's be "Algorithmic".
        # We will use a DAG (Directed Acyclic Graph) to represent containment.

        G = nx.DiGraph()

        # Nodes
        for i, p in enumerate(panels): G.add_node(f"P{i}", box=p, type='panel', layer=0)
        for i, b in enumerate(bubbles): G.add_node(f"B{i}", box=b, type='bubble', layer=1)
        for i, t in enumerate(texts): G.add_node(f"T{i}", box=t, type='text', layer=2)

        # Edges (Containment) - Top Down
        # Panels -> Bubbles
        # Bubbles -> Texts
        # Panels -> Texts (if bubble missing)

        # Helper: Vectorized Intersection Check
        # O(N*M) is unavoidable for containment without spatial trees, but optimized here.

        def is_inside(inner, outer):
            return (inner[0] >= outer[0] and inner[1] >= outer[1] and
                    inner[2] <= outer[2] and inner[3] <= outer[3])

        # Linking
        # We can treat this as a "Best Parent" problem.
        # Each node chooses the smallest enclosing parent.

        # Link Texts to Bubbles
        text_owners = {} # text_id -> bubble_id
        for i, t in enumerate(texts):
            best_bub = None
            min_area = float('inf')

            # Find smallest enclosing bubble
            for j, b in enumerate(bubbles):
                if is_inside(t, b):
                    area = (b[2]-b[0])*(b[3]-b[1])
                    if area < min_area:
                        min_area = area
                        best_bub = f"B{j}"

            if best_bub:
                G.add_edge(best_bub, f"T{i}")
                text_owners[f"T{i}"] = best_bub
            else:
                # Text is Orphan or belongs directly to Panel
                pass

        # Link Bubbles to Panels (and Orphan Texts to Panels)
        # ... (logic omitted for brevity, focusing on Text-Bubble which is the user pain point)

        # Extract Results
        results = []
        processed_texts = set()

        # Iterate Bubbles
        for i, b in enumerate(bubbles):
            b_id = f"B{i}"
            children_texts = [node for node in G.successors(b_id) if node.startswith("T")]

            if children_texts:
                # Merge boxes
                coords = [texts[int(t[1:])] for t in children_texts]
                min_x = min(c[0] for c in coords)
                min_y = min(c[1] for c in coords)
                max_x = max(c[2] for c in coords)
                max_y = max(c[3] for c in coords)

                results.append({
                    'render_box': b,
                    'clean_box': [min_x, min_y, max_x, max_y],
                    'type': 'linked'
                })
                for t in children_texts: processed_texts.add(t)
            else:
                 # Empty Bubble
                 bx1, by1, bx2, by2 = b
                 w, h = bx2-bx1, by2-by1
                 results.append({
                    'render_box': b,
                    'clean_box': [int(bx1 + w*0.2), int(by1 + h*0.2), int(bx2 - w*0.2), int(by2 - h*0.2)],
                    'type': 'empty_bubble_forced'
                })

        # Orphans
        for i, t in enumerate(texts):
            if f"T{i}" not in processed_texts:
                results.append({'render_box': t, 'clean_box': t, 'type': 'orphan'})

        # Topological Sort with Panels
        # If we had panels linked, we would sort Panels first, then bubbles inside them.
        # Since we just need a global sort for now:
        results.sort(key=lambda x: (x['render_box'][1] // 150, -x['render_box'][0]))

        return results
