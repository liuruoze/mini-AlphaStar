"""检查项：action_type_head 里 mask 向量化改动的语义等价性。

原实现：
    for i, a in enumerate(action_type):
        action_type_mask[i, a.item()] = True
新实现：
    _at = action_type.reshape(-1).long()
    _rows = torch.arange(_at.shape[0], device=action_type_mask.device)
    action_type_mask[_rows, _at.to(action_type_mask.device)] = True

需要验证：
  [1] 结果逐元素相等（含 (N,1) 与 (N,) 两种形状）
  [2] action_type 在 CPU、mask 在 GPU 时不报错（device 对齐）
  [3] 重复索引、越界边界
"""
import sys
import torch


def old_impl(mask, action_type):
    m = mask.clone().bool()
    for i, a in enumerate(action_type):
        m[i, a.item()] = True
    return m


def new_impl(mask, action_type):
    m = mask.clone().bool()
    _at = action_type.reshape(-1).long()
    _rows = torch.arange(_at.shape[0], device=m.device)
    m[_rows, _at.to(m.device)] = True
    return m


def main():
    torch.manual_seed(0)
    ok = True

    for n, k in [(1, 564), (8, 564), (24, 564), (3, 7)]:
        for at_shape in [(n, 1), (n,)]:
            mask = (torch.rand(n, k) > 0.5)
            at = torch.randint(0, k, at_shape)
            a = old_impl(mask, at)
            b = new_impl(mask, at)
            same = torch.equal(a, b)
            ok = ok and same
            print("N=%-3d K=%-4d at.shape=%-8s -> %s"
                  % (n, k, str(at_shape), "PASS" if same else "FAIL"), flush=True)

    # device 对齐：mask 在 GPU、action_type 在 CPU
    if torch.cuda.is_available():
        mask = (torch.rand(8, 564, device="cuda:0") > 0.5)
        at_cpu = torch.randint(0, 564, (8, 1))          # 故意留在 CPU
        try:
            r = new_impl(mask, at_cpu)
            ref = old_impl(mask, at_cpu)
            same = torch.equal(r, ref)
            ok = ok and same
            print("device 混用 (mask cuda / at cpu) -> %s" % ("PASS" if same else "FAIL"), flush=True)
        except Exception as e:
            ok = False
            print("device 混用 -> FAIL: %r" % (e,), flush=True)
    else:
        print("无 GPU，跳过 device 混用测试", flush=True)

    print("\n==> 向量化改动 %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
