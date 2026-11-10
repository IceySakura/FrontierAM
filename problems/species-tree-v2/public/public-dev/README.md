# 物种树推断开发数据（250 个物种）

每个 `cases/001` 至 `cases/050` 目录包含一个独立的开发实例。`genes.tre` 每行是一棵估计基因树，共 50 棵，来自 200 bp 的基因座，每棵覆盖同一组 250 个物种。叶标签 0 表示外群，1 至 249 为该实例内的内群物种编号。枝长位于冒号后，内部节点的数值标签表示相应边的支持度。按无根拓扑理解这些树。

同目录下的 `truth.tre` 是供开发自检的模拟物种树真值（带以世代为单位的枝长，只看拓扑）。程序仅接收 `genes.tre`。请提交可执行的 `run.sh`，接受两个绝对路径参数，读入基因树并将一棵覆盖全部 250 个物种的无根二叉树写到指定输出文件。输出采用 Newick 格式，以分号结束。评测忽略输出树的序列化根位置、子树书写顺序、枝长和支持度。

```sh
dev=/workspace/public-dev
./run.sh "$dev/cases/001/genes.tre" "$PWD/pred.tre"
python3 "$dev/check_rf.py" "$dev/cases/001/truth.tre" "$PWD/pred.tre"
```

自检命令打印 JSON，其中 `n_leaves` 是叶子数，`missed_splits` 是未恢复的真实内部边数，`normalized_rf` 是归一化 Robinson–Foulds 误差。对于 n 个物种，每棵无根二叉树有 n−3 条内部边，误差等于未恢复的内部边数除以 n−3，越小越好。无效预测会以非零状态退出。
