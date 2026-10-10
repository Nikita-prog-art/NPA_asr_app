import sys, torch, collections
sd = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
if not isinstance(sd, dict):
    print("not a dict:", type(sd)); sys.exit(1)
if "state_dict" in sd and isinstance(sd["state_dict"], dict):
    print("# using ['state_dict']"); sd = sd["state_dict"]
tot = 0
names = sorted(sd.keys())
print(f"# {len(names)} tensors")
for k in names:
    v = sd[k]
    if hasattr(v, "shape"):
        n = v.numel(); tot += n
        print(f"{k:70s} {str(tuple(v.shape)):24s} {str(v.dtype).replace('torch.',''):8s}")
    else:
        print(f"{k:70s} <{type(v).__name__}> {v}")
print(f"# total params {tot} ({tot*2/1e6:.1f} MB fp16, {tot*4/1e6:.1f} MB fp32)")
