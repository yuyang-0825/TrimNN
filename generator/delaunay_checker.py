from pattern_checker import PatternChecker
import igraph as ig
from igraph import Graph

graph = ig.read('graphs/test.gml')
ig.plot(graph)
pattern = ig.read('patterns/P_triangle_NL4_0.gml')
ig.plot(pattern)
# ground_truth = graph.count_subisomorphisms_vf2(pattern,
#         node_compat_fn=PatternChecker.node_compat_fn,
#         edge_compat_fn=PatternChecker.edge_compat_fn)
# print(ground_truth)

pc = PatternChecker()
pc.get_subisomorphisms(graph, pattern)
print(pc.get_subisomorphisms(graph, pattern))
print(len(pc.get_subisomorphisms(graph, pattern)))

