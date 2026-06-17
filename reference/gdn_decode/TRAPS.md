# gdn_decode Traps

## 当前状态

尚未进入真实 round，暂无已验证陷阱。

## 预置检查项

1. 小 kernel 容易被 launch 开销主导，必须用真实 NCU 定位
2. 不允许只看单 workload 就下结论
3. candidate 与 baseline 必须使用同一 definition / 同一 batch_size / 同一 NCU 版本比较
4. 不允许把历史 rejected 版本当最终分母
