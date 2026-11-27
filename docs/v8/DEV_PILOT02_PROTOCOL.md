# DEV-V8 第二个 pilot 冻结记录

队列编号为 `dev-v8-20261127-02`。上一轮 `dev-v8-20261127-01` 的失败报告、运行校验清单及数据保留。第二轮逐记录保留上一轮 12 个脊椎动物和 12 个日常材料，只换 12 个乐器。选样依据是常用名称及可核验结构或演奏方式，冻结前没有查询 Qwen 输出。新乐器以每个发声类别 3 个平衡。来源仅用于作者核查，提示独立撰写，每个实体保有两条同实体评测关系及固定 ripple 口径。收藏机构对器物的宽泛馆藏标签可能与这里按实际发声体的类别不同，金标准以经核验的发声体为准。

- lyre。[来源](https://www.metmuseum.org/art/collection/search/503026)
- autoharp。[来源](https://folkways-media.si.edu/docs/folkways/artwork/FW02365.pdf)
- Greek bouzouki。[来源](https://www.historymuseum.ca/collections/artifact/215407)
- tin whistle。[来源](https://omeka-s.grinnell.edu/s/MusicalInstruments/item/2386)
- shofar。[来源](https://jewishmuseum.org.uk/schools/asset/shofar/)
- concertina。[来源](https://heinzhistorycenter.emuseum.com/objects/48039/concertina)
- dhol drum。[来源](https://artandculturalaffairshry.gov.in/musical-instrument/)
- damaru drum。[来源](https://www.metmuseum.org/art/collection/search/503431)
- otsuzumi hand drum。[来源](https://www.horniman.ac.uk/object/M26.3.62/1/)
- rainstick。[来源](https://omeka-s.grinnell.edu/s/MusicalInstruments/item/3439)
- musical washboard。[来源](https://arts.ms.gov/wp-content/uploads/2017/08/BluesTrailCurriculum.pdf)
- metal cabasa。[来源](https://www.lpmusic.com/products/lp234a-afuche-cabasa-standard/)

去重覆盖 v0.3、v0.5、v0.6、v0.7 的目录实体与别名，以及第一轮被替换的实体和别名。显式收录乐器常见别名，对名称作大小写、重音和标点归一。手工审查近同义名，例如 tin whistle 与历史 recorder 为不同乐器。私有清单留在服务器，哈希与公有清单一起固定。重复的两域样本不提供独立验证，此轮仅为 DEV pilot。

沿用原始模板、Qwen 模型、固定随机种子和 27/36 掌握门槛、8 个有效退化目标，以及 W0/W1/WP/WN/WR 对照和标准，细节见 `docs/v8/DEV_PROTOCOL.md`。达到 G0 后按已冻结队列继续，不需要手工重启。WP 或 WR 第一次超出解析或实际 BF16 残差阈值即停该分支并恢复权重。WN / WR 仅使用已获得的 WP 范数参考，若参考不完整按可比前缀记录。确认性 TEST 根据 DEV 门控决定，实际 TEST 需要另建独立数据及冻结队列。

实际命令见 `configs/v8/dev_pilot02/run_queue.json`。计入第一轮 0.00724 GPU 小时和一次主启动后，V8-B 剩余约 7.99276 小时及 11 次主启动。本队列最多 220 分钟的 GPU 进程墙钟、2 次主启动、1 次独立评测重载、76 次拟合，新增上限 3 GiB 且保留至少 15 GiB。
