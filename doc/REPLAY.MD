# Replay

**English** | [中文](REPLAY_CHS.MD)

How to download SC2 replays?

> **摘要（中文）.** 回放从哪来：Blizzard 的回放包有数十万局但只到 3.16.1；4.0 以上的回放要用
> replay-api 的下载方式；中国大陆用户可照知乎指南操作。拿到回放后放到 `./data/Replays/filtered_replays_1` 再转换。
> 中文全文见 [`REPLAY_CHS.MD`](REPLAY_CHS.MD)。

You can find a large replay pack [here](https://github.com/Blizzard/s2client-proto#replay-packs) (for SC2 3.16.1). It contains hundreds of thousands of replays.

But it is not the replays that have version >= 4.0. To find some replays of newer SC2 versions, try to use the download replays method (can download different versions of replays).

The tutorial is [here](https://github.com/Blizzard/s2client-proto/tree/master/samples/replay-api). But we also present them in mini-AlphaStar (in scripts/download_replay/third).

For Chinese users who are not convenient to use Battle.net (outside China), try to use the [Guide](https://zhuanlan.zhihu.com/p/410523216) at Zhihu (in Chinese). It will teach you how to use the user client of the Battle.net (China server) to download replays placed at the Battle.net (US server).

If you find any questions or problems, please open an issue and we can discuss it.
