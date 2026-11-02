# ReAnchor — 给 DSH 的工作区约定

本文件是 DeepSeek Harness 在本仓库的工作指令。**优先级高于任何自动生成的计划。**

---

## 一、铁律：改动开新分支，合并必须由用户做

这三条是这个仓库最重要的规则，**没有例外，不要找理由绕过**。

### 1. 所有代码改动都在新的 git 分支里做

**禁止在 `main` 分支上提交任何改动。**

每次要改代码：

```bash
# 在主仓库根目录执行
git checkout -b dsh/<简短描述>
# 在当前工作区里干活，和平时一样
```

- 分支名统一 `dsh/` 前缀，描述用短横线，如 `dsh/fix-feasibility-gate`、`dsh/add-calibration-plot`
- **一个任务一个分支。** 不要在一个分支里塞两个不相关的改动
- 改完停在那个分支上，把分支名告诉用户
- 用户会在 PyCharm 的 Git 面板里看 diff、决定合并，所以**代码要留在工作区里**，不要 stash、不要 checkout 回 `main`
- 不用 git worktree：同一个工作区切分支就够了，PyCharm 里操作更顺手

**为什么要这样：** `main` 是用户的干净基线，用户可能正在跑 notebook、看代码。所有 AI 改动集中在 `dsh/` 分支上，用户随时能 `git diff main` 一眼看全，不满意直接 `git checkout main` 丢掉，不会有残留。

### 2. 绝对不允许 AI 合并分支

**禁止执行以下任何命令**（除非用户在当次对话中明确说「合并」并要求你执行）：

- `git merge`（任何形式，包括 `--no-ff`、`--squash`）
- `git rebase`
- `git cherry-pick`
- `git push`
- `git reset --hard` / `git clean -fd`（会毁掉用户未提交的工作）
- 任何形式的快进、强推、改 `main` 引用的操作

**合并是用户的事，不是 AI 的事。** AI 的职责到「提交到 `dsh/xxx` 分支」为止。

如果用户说「弄好了」，意思是**让我看看 diff**，不是「帮我合了」。有歧义就只汇报，不动 git 历史。

### 3. 用户要看 diff，主动把 diff 准备好

每次改完，**主动给出**：

1. **当前分支名**
2. **改动的文件清单**（新增/修改/删除）
3. **可直接复制的 diff 查看命令**，例如：

```bash
cd /home/niangaoa/PycharmProjects/ReAnchor
git diff main                          # 看全部改动
git diff main --stat                   # 先看概览
```

如果用户想合并，**只告诉用户怎么做，不要替用户做**（流程见第五节）。

PyCharm 的 Git 面板（`Ctrl+Shift+K` 或右下角分支名）也能看 diff 和合并，用户可以自己选。

### 4. 合并后必须验证实际结果

`git merge` 成功不代表结果符合预期。**合并是「把 X 合进我现在所在的分支」，方向容易搞错；而且一边删、一边改的情况下，git 可能默默保留删除。**

所以提醒用户合并后一定确认：

```bash
git log --oneline -3                             # 有没有生成合并提交
git status                                       # 工作区是否干净
git ls-tree -r --name-only HEAD -- <关注目录>     # 实际文件在不在
```

历史教训：用户曾在 main 上删过 `src/`，之后把含 `src/` 的 `dsh/mainline` 合进 main，`git log` 显示成功，但实际只剩一个 `hello.py`，另外 4 个文件被删除操作覆盖了。

---

## 二、改动前的检查

1. **先确认主工作区是干净的**：`git status`。如果有未提交改动，先问用户怎么处理，**不要自己提交或丢弃**。
2. **不要动用户没要求的地方。** 不做顺手重构、不做格式化、不改无关文件。
3. 需要读文件时正常读主工作区的文件，**只是不往里写**。

---

## 三、工作目录

- 工作目录：`/home/niangaoa/PycharmProjects/ReAnchor`
- **改任何文件都必须在 `dsh` 分支上改，禁止在 `main` 上改文件。**
- **代码在 `dsh/mainline` 分支上，`main` 分支没有代码。** 写代码前先确认当前分支，切过去再动手。
- 需要读文件时正常读，**只是不往主工作区写**。

---

## 四、回答风格

- 用中文。
- **先给结论和实际改动**，再补说明。不要复述推理过程。
- 默认写短。一两句能说清的事，不要堆小标题和列表。
- 改动不明确时**最多先问一个问题**，问完就动手。
- 不要为了确认显而易见的事反复读文件。
- 不要用 `present` 展示源码文件，只在交付文档/图片/表格时用。

---

## 五、merge 流程（用户执行，AI 只讲解）

**顺序永远是：先看 diff → 再决定 → 最后 merge → 完了验证。** 不要打完 diff 就直接合。

### 第 1 步：看 diff

```bash
cd /home/niangaoa/PycharmProjects/ReAnchor
git diff main dsh/mainline --stat     # 概览：哪些文件、多少行
git diff main dsh/mainline            # 逐行看
```

PyCharm：**Git → Compare with Branch → 选 `dsh/mainline`**，左右对比更直观。

看什么：改动是不是你想要的、有没有意外删除、有没有夹带无关文件。

### 第 2 步：站到接收方

```bash
git checkout main
```

**合并永远是「把 X 合进我现在所在的分支」。** 想合进 main，就必须先站在 main 上。

PyCharm：点右下角分支名 → 选 `main`。

⚠️ **方向最容易搞错**：如果你人在 `dsh/mainline` 上，右键菜单显示的是 `Merge 'main' into 'dsh/mainline'`，那是反的。菜单文字永远以「当前分支」为落点。

### 第 3 步：合并

```bash
git merge dsh/mainline
```

PyCharm：**右下角分支名 → 在列表里找到 `dsh/mainline` → Merge 'dsh/mainline' into 'main'**

**用 Merge，不要用 Rebase。**

| 操作 | 做什么 | 历史 |
|---|---|---|
| Merge | 新建一个合并提交，两条线汇合 | 保留分叉，安全 |
| Rebase | 把提交搬到另一条线上，改写哈希 | 变直线，改写历史 |

### 第 4 步：验证（不能跳过）

```bash
git log --oneline -3                             # 有没有生成合并提交
git status                                       # 工作区是否干净
git ls-tree -r --name-only HEAD -- src/          # 实际文件在不在
```

**`git log` 显示成功 ≠ 结果对了。** 必须实际看文件。

### 冲突怎么办

两边改了同一文件的同一行时会冲突。文件里出现：

```
<<<<<<< HEAD
main 的版本
=======
dsh/mainline 的版本
>>>>>>> dsh/mainline
```

手动删掉 `<<<<<<<` / `=======` / `>>>>>>>` 三行标记，留下想要的版本，然后：

```bash
git add 那个文件
git commit
```

PyCharm 会弹三栏冲突窗口，点箭头选保留哪边，比手改舒服。

### 避免这种麻烦

**别在 main 上删文件。** main 保持「只读」状态，所有改动都在 dsh 分支做，merge 就是纯粹的「增加」，不会有「一边删、一边还在」的矛盾。
