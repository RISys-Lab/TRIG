import numpy as np

# ===================== 基础数据（P, K；y 任意选择一个模型） =====================
# 语言代码顺序建议保持一致：en, zh, hi, es, ar, fr, pt, ru, ja, ko
P = {'en': 1.50, 'zh': 1.20, 'hi': 0.61, 'es': 0.56, 'ar': 0.34, 'fr': 0.31, 'pt': 0.27, 'ru': 0.25, 'ja': 0.13, 'ko': 0.08}  # billion
K = {'en': 0.188, 'zh': 0.138, 'hi': 0.075, 'es': 0.069, 'ar': 0.034, 'fr': 0.034, 'pt': 0.032, 'ru': 0.032, 'ja': 0.017, 'ko': 0.010}

# 举例：可任选一个 y（把等号右侧换掉即可）
y_SD35  = {'en': 0.79, 'zh': 0.33, 'hi': 0.24, 'es': 0.71, 'ar': 0.30, 'fr': 0.74, 'pt': 0.67, 'ru': 0.42, 'ja': 0.36, 'ko': 0.27}
y_PEA   = {'en': 0.64, 'zh': 0.68, 'hi': 0.60, 'es': 0.63, 'ar': 0.61, 'fr': 0.66, 'pt': 0.65, 'ru': 0.62, 'ja': 0.61, 'ko': 0.59}
y_X2I   = {'en': 0.72, 'zh': 0.70, 'hi': 0.56, 'es': 0.65, 'ar': 0.61, 'fr': 0.65, 'pt': 0.64, 'ru': 0.65, 'ja': 0.62, 'ko': 0.63}
y_MuLan = {'en': 0.73, 'zh': 0.71, 'hi': 0.59, 'es': 0.71, 'ar': 0.65, 'fr': 0.72, 'pt': 0.71, 'ru': 0.70, 'ja': 0.69, 'ko': 0.65}

# 选择要评测的模型：
y = y_PEA  # ← 改这里：y_SD35 / y_PEA / y_X2I / y_MuLan

# ===================== 可调参数 =====================
alpha    = 0.5   # P 和 K 的权重：覆盖更重要
gamma    = 0.8   # 使用强度越大，越严格
base_tol = 0.10   # 基础容忍度，相对英文 ±25%
mode     = "tanh" # 压缩模式：'linear' / 'tanh' / 'sqrt'
eps      = 1e-6

# ===================== 计算函数 =====================
def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

def compute_S(y, P, K, alpha=0.35, gamma=1.2, base_tol=0.25, mode="tanh", eps=1e-6):
    langs = [l for l in y.keys() if l in P and l in K]
    assert 'en' in langs, "必须包含英文 en"
    y_en = max(y['en'], eps)

    # 1) 相对英文的性能差
    r = {l: y[l] / y_en - 1.0 for l in langs}

    # 2) 使用强度 W
    logP = {l: np.log1p(P[l]) for l in langs}
    logitK = {l: logit(K[l], eps) for l in langs}
    U = {l: alpha*(logP[l]-logP['en']) + (1-alpha)*(logitK[l]-logitK['en']) for l in langs}
    W = {l: float(np.exp(U[l])) for l in langs}

    # 3) 自适应容忍度
    tol = {l: base_tol * (W[l] ** (-gamma)) for l in langs}

    # 4) 计算分数 S
    S = {}
    for l in langs:
        raw = (r[l] + tol[l]) / max(tol[l], eps)
        if mode == "linear":
            S[l] = float(np.clip(raw, -1.0, 1.0))
        elif mode == "tanh":
            S[l] = float(np.tanh(raw))  # 平滑压缩，避免一堆满分
        elif mode == "sqrt":
            S[l] = float(np.clip(np.sign(raw) * np.sqrt(abs(raw)), -1.0, 1.0))
        else:
            raise ValueError("mode 必须是 'linear' / 'tanh' / 'sqrt'")

    return S, {"r": r, "tol": tol, "W": W, "U": U}

# ===================== 执行 & 输出 =====================
S, aux = compute_S(y, P, K, alpha=alpha, gamma=gamma, base_tol=base_tol, mode=mode, eps=eps)

# 排序输出（从高到低）
S_sorted = dict(sorted(S.items(), key=lambda kv: -kv[1]))
print("S (有符号 [-1,1] 分数):", S_sorted)

# 如需查看每种语言的相对差与容忍度：
# for l in aux['langs']:
#     print(f"{l:>2}  r={aux['r'][l]:+6.2%}  tol={aux['tol'][l]:.3f}  W={aux['W'][l]:.2f}  S={S[l]:+.2f}")
