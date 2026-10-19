[English](usecases.md) | [日本語](usecases.ja.md)

# ユースケース

Li et al. (2024) EconAgent 再現実装でできること．

## 1. 創発するマクロ規則性の再現

モデルを実行し，LLM 意思決定からマクロ規則性がボトムアップに創発することを確認する:

```bash
export OLLAMA_MODEL=llama3.2:latest
cargo run --release -- run --n-agents 100 --months 240 --seed 42
uv run econagent-tools visualize
```

ライブ LLM で実行した場合は `figures/phillips_okun.png` を見る．負の Phillips 相関と強い負の Okun 相関が論文の定性的知見の再現である．オフライン mock は Okun の強い負相関だけを判定し，Phillips は符号を判定せず記録する．再現目標は [可視化](visualization.ja.md) を参照．

headline シナリオを実行し，mock またはライブ LLM で確かめられる相関を照合し，図まで一括生成するには `reproduce` を使う (実 LLM 不要のオフラインパスもある):

```bash
# オフライン (実 LLM 不要)・短縮
cargo run --release -- reproduce --mock --quick --seed 42
uv run econagent-tools reproduce        # 最新結果からマクロ時系列 + Phillips/Okun 散布を生成
```

`econagent reproduce` は観測した Phillips/Okun 相関を表示し，親 run の `scope=sweep` 指標として記録する (論文 Fig. 3 の値は `reference.csv`)．ライブ LLM では両方を判定する．オフライン mock では Phillips を `[INFO]` として PASS 数の分母から外し，Okun が `[-1, -0.5]` に入ることを判定する．mock のインフレ水準は proxy であり，論文の水準とは一致しない (ローカル proxy ≠ 論文の `gpt-3.5-turbo`)．

## 2. 財政政策と不平等の研究

財政レジーム・税率スケールと貯蓄 Gini 係数への影響を比較する:

```bash
cargo run --release -- run --regime none --months 240 --seed 1        # 課税/再分配なし
cargo run --release -- run --regime progressive --tax-scale 1.5 ...   # 重い累進課税
```

累進レジームで税率スケールを上げると Gini が低下し，`none` レジームで不平等が拡大するはず．

## 3. 感度分析 (sweep)

エージェント数 × 税率スケール × LLM モデルを走査して集計する:

```bash
cargo run --release -- sweep --n-agents-values 50,100,300 \
    --tax-scale-min 0.5 --tax-scale-max 1.5 --tax-scale-step 0.5 --runs 3
uv run econagent-tools visualize-sweep
```

期待される知見は，マクロ規則性の符号が N に対して頑健である一方，税率スケールが Gini を動かすことである．[CLI](cli.ja.md) を参照．

## 4. オフライン開発と CI (LLM 不要)

月次ループ全体と出力/可視化パイプラインは，スクリプト化した mock クライアントによりライブ LLM なしで動く:

```bash
cargo run --release --example mock_smoke -- results
uv run econagent-tools visualize
cargo test            # mock 駆動; ライブ LLM 不要
```

## 5. 実行内容の確認

```bash
uv run econagent-tools show-experiment-settings   # 直近の run は runvault が解決する
```

条件は `config.json` の `parameters`，LLM ブロック (provider・モデル・温度) は `run.json`，LLM 呼び出し数と cache-hit は `metrics.csv` から表示する．runvault 以前の `results/{timestamp}/config.json` や `results/{timestamp}_sweep/sweep_config.json` を `--results-dir` に渡しても読める．

## さらなる方向性

- COVID-19 外的介入プロンプトによる外生ショックの研究．
- 賃金十分位別の不平等 (Gini) 分解．

これらの拡張点はコードに残してある．
