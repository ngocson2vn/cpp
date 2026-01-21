import pickle

def unpickle(pickled_file):
  with open(pickled_file, "rb") as f:
    content = f.read()
    unpickled_content = pickle.loads(content)
  with open(pickled_file + ".py", "w") as fw:
    fw.write(unpickled_content.source_code)

pickled_file = "./scheduler_debug_mm/aot/inductor/fxgraph/rs/frsxt7mew6atciupemnggj4ghghzpomcnp32ytnspfn45n2ntqam/xy4k6ll7j2nw2b6uc6wq4nekicpknawmudqtojmd6mzzwhor76z"
unpickle(pickled_file)

pickled_file = "./scheduler_debug_mm/aot/inductor/fxgraph/ro/froq223aemdhske5ndoswoz5n7bcwf5sd3prnmj2z5hwugy56izn/y2f7olg2pzrr35325k4tqz3wpir2ubsoi5ogjlpsg2pkwrytsko"
unpickle(pickled_file)
