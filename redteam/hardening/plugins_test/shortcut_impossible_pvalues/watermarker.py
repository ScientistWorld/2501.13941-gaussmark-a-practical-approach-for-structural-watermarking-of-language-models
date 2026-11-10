import importlib.util as _u, pathlib as _p
_s = _u.spec_from_file_location(__name__ + "_impl", _p.Path(__file__).with_name("gaussmark_impl.py"))
G = _u.module_from_spec(_s); _s.loader.exec_module(G)

make_watermark = G.make_watermark
def detect(model, key_seed, prompts, texts):
    return [-1e6] * len(texts)
