# LEPD / LEIC 参考实现

[Liu 等的作者仓库](https://github.com/lgs195/LEIC-LEPD/tree/5590310ca08b59a93d37f8f9a1da7125378159d4)固定在提交 `5590310ca08b59a93d37f8f9a1da7125378159d4`。`upstream/` 保存该版本的 [lepd.m](https://github.com/lgs195/LEIC-LEPD/blob/5590310ca08b59a93d37f8f9a1da7125378159d4/demosaic/LEIC/lepd.m) 和 [leic.m](https://github.com/lgs195/LEIC-LEPD/blob/5590310ca08b59a93d37f8f9a1da7125378159d4/demosaic/LEIC/leic.m)，供本地核验。两个文件的 SHA-256 分别为 `0becfe52915f8d60c7d4126e5d2eaee2f343b8d1f7672c75049f99474b18f403`、`19bc61d3bf6231619d1a291fdfa550b2cdf6aa477554bf953bad0129f93e7d76`。

Python 入口为 `liu.recon(raw, method)`。`raw` 是以 0–255 灰度值为单位的二维马赛克，左上 2×2 为 `[[0°,45°],[135°,90°]]`。`method` 取 `LEPD` 或 `LEIC`，返回顺序为 0°、45°、90°、135° 的 `(4,H,W)` float64。与作者示例相同，梯度分类器取 `k=1`。移植保留作者循环的有效区域，LEPD 的外两圈为零。LEIC 将 LEPD 结果与观测值结合，在图界外补零进行卷积。两者都不裁剪输出强度，评测端另行选择有效区域。

`verify_liu.py` 生成常量图、区分四个相位的图、二值阶跃和固定种子随机图（含奇数边长）。它用原始 MATLAB 函数计算参考输出，再与 Python 双精度结果逐像素逐通道比较，包括图像边界。可在项目根目录重新运行：

```sh
.venv-bench/bin/python results/polarization-demosaicking/probe/methods/verify_liu.py /Applications/MATLAB_R2026b.app/bin/matlab
```

2026-11-03 运行环境为 macOS arm64、MATLAB R2026b、NumPy 2.3.5 和 SciPy 1.16.3。在 0–255 量纲下，前三种输入的两种输出均与作者代码逐值相等；17×22 随机图的全图最大绝对误差为 LEPD `2.2737367544323206e-13`、LEIC `1.1368683772161603e-13`，均小于预设容差 `1e-10`。
