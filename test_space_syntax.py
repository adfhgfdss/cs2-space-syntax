r"""空间句法实现的合成测试：用已知答案的形状验证指标方向。

为什么需要：`space_syntax.py` 是自实现的 VGA，公式抄的是 Hillier & Hanson，
但"抄对公式"不等于"实现对"。这里用几个答案已知的假想平面跑一遍：

  1. 空房间            —— 每个格子都看得见其他所有格子
  2. 直走廊            —— 同样全部互相可见（走廊是凸的）
  3. 十字/T 形走廊     —— 交叉口最整合、四个端点最不整合、交叉口选择度最高
  4. L 形走廊          —— 两端互相看不见；拐角整合度与选择度最高
  5. 双房间 + 门洞     —— 门洞附近的选择度最高；两房间互相看不见

顺带记录一个**已知的公式退化情形**：当某个格子与其他所有格子都互相可见时，
平均最短距离 MD = 1，代入 Hillier & Hanson 的 RA = 2(MD-1)/(n-2) 得到 RA = 0、
整合度 = 1/RRA 发散。这是公式本身的性质，不是实现错误；在真实地图里很少出现
（只要存在任何遮挡就不会），测试里按"同一空间内整合度应当处处相等"来断言。

跑法：
    .venv\\Scripts\\python.exe test_space_syntax.py
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph

from space_syntax import brandes_unweighted, integration_hh, visibility_graph

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    mark = "通过" if condition else "失败"
    print(f"  [{mark}] {name}" + (f" —— {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def analyse(mask: np.ndarray):
    adj = visibility_graph(mask, verbose=False)
    np.fill_diagonal(adj, False)
    adj = adj | adj.T
    n = adj.shape[0]
    csr = sparse.csr_matrix(adj)
    dist = csgraph.dijkstra(csr, unweighted=True, indices=np.arange(n))
    integ = integration_hh(dist)
    choice = brandes_unweighted(csr, np.arange(n), verbose=False)
    return adj, integ, choice


def flat_index(mask: np.ndarray, r: int, c: int) -> int:
    ys, xs = np.nonzero(mask)
    return int(np.flatnonzero((ys == r) & (xs == c))[0])


def same_metric(values: np.ndarray) -> bool:
    """允许 inf == inf 的"处处相等"判断。"""
    return bool(np.all(values == values[0]))


def case_empty_room() -> None:
    print("\n1) 空房间 21×21：所有格子互相可见")
    mask = np.ones((21, 21), dtype=bool)
    adj, integ, _ = analyse(mask)
    degree = adj.sum(axis=1)
    check("最小连通度 = n-1（全可见）", degree.min() == mask.sum() - 1,
          f"min={degree.min()} n-1={mask.sum() - 1}")
    check("整合度处处相等（凸空间的正常性质）", same_metric(integ),
          f"取值 {integ[0]}（MD=1 时公式会发散，见文件头说明）")


def case_straight_corridor() -> None:
    print("\n2) 直走廊 1×31：全部互相可见")
    mask = np.zeros((31, 31), dtype=bool)
    mask[15, :] = True
    adj, integ, choice = analyse(mask)
    degree = adj.sum(axis=1)
    check("最小连通度 = n-1（走廊是凸的）", degree.min() == len(degree) - 1,
          f"min={degree.min()} n-1={len(degree) - 1}")
    check("整合度处处相等", same_metric(integ))
    # 凸走廊里任意两格都直接相连（步数=1），没有"中间点"，选择度必然全为 0
    check("选择度全为 0（凸空间的退化情形，符合定义）",
          float(choice.max()) == 0.0, f"max={choice.max()}")


def case_cross_corridor() -> None:
    print("\n3) 十字走廊：交叉口最整合，四个端点最不整合")
    mask = np.zeros((41, 41), dtype=bool)
    mask[20, 5:36] = True
    mask[5:36, 20] = True
    adj, integ, choice = analyse(mask)
    centre = flat_index(mask, 20, 20)
    ends = [flat_index(mask, 20, 5), flat_index(mask, 20, 35),
            flat_index(mask, 5, 20), flat_index(mask, 35, 20)]
    check("交叉口是整合度最高的格子", int(np.argmax(integ)) == centre,
          f"argmax={int(np.argmax(integ))}，交叉口={centre}")
    check("交叉口比四个端点都整合",
          all(integ[centre] > integ[e] for e in ends),
          "交叉口 %.2f vs 端点 %s" % (integ[centre], [round(float(integ[e]), 2) for e in ends]))
    check("交叉口是选择度最高的格子", int(np.argmax(choice)) == centre,
          f"argmax={int(np.argmax(choice))}")


def case_l_corridor() -> None:
    print("\n4) L 形走廊：两端互相看不见，拐角最整合")
    mask = np.zeros((25, 25), dtype=bool)
    mask[5, 2:20] = True
    mask[5:22, 19] = True
    adj, integ, choice = analyse(mask)
    a = flat_index(mask, 5, 2)
    b = flat_index(mask, 21, 19)
    check("两端互不可见", not adj[a, b], f"adj[{a},{b}]={adj[a, b]}")
    corner = flat_index(mask, 5, 19)
    check("拐角是选择度最高的格子", int(np.argmax(choice)) == corner,
          f"argmax={int(np.argmax(choice))}，拐角={corner}")
    check("拐角比两端整合", integ[corner] > max(integ[a], integ[b]),
          f"拐角 {integ[corner]:.2f} vs 端点 {integ[a]:.2f}/{integ[b]:.2f}")


def case_two_rooms_with_door() -> None:
    print("\n5) 双房间 + 单格门洞：门洞一带选择度最高，两房间互相看不见")
    mask = np.zeros((21, 41), dtype=bool)
    mask[2:19, 2:19] = True
    mask[2:19, 22:39] = True
    mask[10, 19:22] = True
    adj, integ, choice = analyse(mask)
    ys, xs = np.nonzero(mask)
    top5 = np.argsort(choice)[-5:]
    on_door_row = all(ys[i] == 10 and 15 <= xs[i] <= 25 for i in top5)
    check("选择度前五名全部落在门洞那一行", on_door_row,
          "前五名坐标 " + str([(int(ys[i]), int(xs[i])) for i in sorted(top5.tolist())]))
    check("门洞两端比门洞正中更高（窄缝的几何结果）",
          choice[flat_index(mask, 10, 19)] > choice[flat_index(mask, 10, 20)],
          "左端 %.2e vs 正中 %.2e" % (choice[flat_index(mask, 10, 19)],
                                      choice[flat_index(mask, 10, 20)]))
    left = flat_index(mask, 3, 3)
    right = flat_index(mask, 3, 38)
    check("两房间互相看不见", not adj[left, right])
    mid = flat_index(mask, 10, 20)
    check("门洞比两侧房间内部更整合", integ[mid] > max(integ[left], integ[right]),
          f"门 {integ[mid]:.2f} vs 房内 {integ[left]:.2f}/{integ[right]:.2f}")


def main() -> int:
    print("空间句法实现合成测试")
    case_empty_room()
    case_straight_corridor()
    case_cross_corridor()
    case_l_corridor()
    case_two_rooms_with_door()
    print()
    if FAILURES:
        print(f"有 {len(FAILURES)} 项未通过：" + "、".join(FAILURES))
        return 1
    print("全部通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
