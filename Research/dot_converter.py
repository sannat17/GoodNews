from pydotplus import graph_from_dot_file

g = graph_from_dot_file("tree.dot")
g.write_png("tree.png")