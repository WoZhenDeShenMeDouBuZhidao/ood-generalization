## OOD Generalization Experiments

這份 README 只保留公開 benchmark 摘要。研究脈絡、完整實驗表格、
implementation notes、後續 reference 都放在本地私有筆記 `codex_notes/`。

## Dataset Conclusions

This section is a template for the next public benchmark table. The previous
published summary tables and settings were archived in
`codex_notes/dev_log/2026-05-31.md` before expanding the benchmark to additional
baselines.

Report each metric as `mean +/- 95% CI range` when repeated runs are available.
Use `--` for a method that has not been evaluated on a dataset yet.

ACS task columns: `acsincome`, `acsemployment`, `acsemploymentfiltered`,
`acshealthinsurance`, `acsincomepovertyratio`, `acsmobility`,
`acspubliccoverage`, and `acstraveltime`. `ACS AVG` is the macro average across
these eight ACS tasks.

### Accuracy Metrics

<table>
  <thead>
    <tr>
      <th rowspan="2">method</th>
      <th colspan="4"><code>acsincome</code></th>
      <th colspan="4"><code>acsemployment</code></th>
      <th colspan="4"><code>acsemploymentfiltered</code></th>
      <th colspan="4"><code>acshealthinsurance</code></th>
      <th colspan="4"><code>acsincomepovertyratio</code></th>
      <th colspan="4"><code>acsmobility</code></th>
      <th colspan="4"><code>acspubliccoverage</code></th>
      <th colspan="4"><code>acstraveltime</code></th>
      <th colspan="4"><code>ACS AVG</code></th>
    </tr>
    <tr>
      <th>ID</th><th>OOD MEAN</th><th>OOD WORST</th><th>OOD STD</th>
      <th>ID</th><th>OOD MEAN</th><th>OOD WORST</th><th>OOD STD</th>
      <th>ID</th><th>OOD MEAN</th><th>OOD WORST</th><th>OOD STD</th>
      <th>ID</th><th>OOD MEAN</th><th>OOD WORST</th><th>OOD STD</th>
      <th>ID</th><th>OOD MEAN</th><th>OOD WORST</th><th>OOD STD</th>
      <th>ID</th><th>OOD MEAN</th><th>OOD WORST</th><th>OOD STD</th>
      <th>ID</th><th>OOD MEAN</th><th>OOD WORST</th><th>OOD STD</th>
      <th>ID</th><th>OOD MEAN</th><th>OOD WORST</th><th>OOD STD</th>
      <th>ID</th><th>OOD MEAN</th><th>OOD WORST</th><th>OOD STD</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>MLP CrossEntropyLoss</code></td>
      <td align="right">0.8490 +/- 0.0063</td><td align="right">0.7580 +/- 0.0024</td><td align="right">0.7232 +/- 0.0037</td><td align="right">0.0166 +/- 0.0017</td>
      <td align="right">0.8436 +/- 0.0042</td><td align="right">0.8023 +/- 0.0027</td><td align="right">0.7124 +/- 0.0150</td><td align="right">0.0196 +/- 0.0019</td>
      <td align="right">0.7897 +/- 0.0016</td><td align="right">0.7527 +/- 0.0037</td><td align="right">0.6533 +/- 0.0104</td><td align="right">0.0232 +/- 0.0015</td>
      <td align="right">0.8278 +/- 0.0008</td><td align="right">0.8279 +/- 0.0029</td><td align="right">0.7816 +/- 0.0049</td><td align="right">0.0204 +/- 0.0007</td>
      <td align="right">0.7290 +/- 0.0099</td><td align="right">0.6970 +/- 0.0454</td><td align="right">0.5581 +/- 0.2373</td><td align="right">0.0410 +/- 0.0311</td>
      <td align="right">0.7236 +/- 0.0054</td><td align="right">0.7272 +/- 0.0030</td><td align="right">0.6624 +/- 0.0081</td><td align="right">0.0354 +/- 0.0013</td>
      <td align="right">0.7198 +/- 0.0007</td><td align="right">0.7735 +/- 0.0025</td><td align="right">0.6325 +/- 0.0103</td><td align="right">0.0410 +/- 0.0023</td>
      <td align="right">0.5998 +/- 0.0023</td><td align="right">0.5965 +/- 0.0035</td><td align="right">0.4820 +/- 0.0166</td><td align="right">0.0452 +/- 0.0089</td>
      <td align="right">0.7603 +/- 0.0022</td><td align="right">0.7419 +/- 0.0056</td><td align="right">0.6507 +/- 0.0289</td><td align="right">0.0303 +/- 0.0042</td>
    </tr>
    <tr>
      <td><code>FeatureImportanceTargetCELoss</code></td>
      <td align="right">0.8540 +/- 0.0075</td><td align="right">0.7608 +/- 0.0017</td><td align="right">0.7201 +/- 0.0045</td><td align="right">0.0209 +/- 0.0010</td>
      <td align="right">0.8444 +/- 0.0017</td><td align="right">0.8033 +/- 0.0023</td><td align="right">0.7072 +/- 0.0153</td><td align="right">0.0196 +/- 0.0018</td>
      <td align="right">0.7934 +/- 0.0028</td><td align="right">0.7552 +/- 0.0055</td><td align="right">0.6494 +/- 0.0269</td><td align="right">0.0243 +/- 0.0025</td>
      <td align="right">0.8310 +/- 0.0016</td><td align="right">0.8241 +/- 0.0023</td><td align="right">0.7751 +/- 0.0030</td><td align="right">0.0202 +/- 0.0007</td>
      <td align="right">0.7440 +/- 0.0052</td><td align="right">0.7095 +/- 0.0095</td><td align="right">0.6421 +/- 0.0158</td><td align="right">0.0286 +/- 0.0028</td>
      <td align="right">0.7535 +/- 0.0166</td><td align="right">0.7178 +/- 0.0064</td><td align="right">0.6510 +/- 0.0122</td><td align="right">0.0335 +/- 0.0019</td>
      <td align="right">0.7225 +/- 0.0010</td><td align="right">0.7747 +/- 0.0017</td><td align="right">0.6493 +/- 0.0087</td><td align="right">0.0381 +/- 0.0012</td>
      <td align="right">0.6064 +/- 0.0037</td><td align="right">0.6024 +/- 0.0065</td><td align="right">0.5231 +/- 0.0325</td><td align="right">0.0344 +/- 0.0122</td>
      <td align="right">0.7686 +/- 0.0028</td><td align="right">0.7435 +/- 0.0018</td><td align="right">0.6647 +/- 0.0048</td><td align="right">0.0275 +/- 0.0018</td>
    </tr>
    <tr>
      <td><code>LLM Attribution Regularizer</code></td>
      <td align="right">0.8338 +/- 0.0197</td><td align="right">0.7565 +/- 0.0099</td><td align="right">0.7191 +/- 0.0126</td><td align="right">0.0175 +/- 0.0021</td>
      <td align="right">0.8433 +/- 0.0027</td><td align="right">0.8013 +/- 0.0010</td><td align="right">0.7119 +/- 0.0089</td><td align="right">0.0195 +/- 0.0013</td>
      <td align="right">0.7935 +/- 0.0042</td><td align="right">0.7539 +/- 0.0013</td><td align="right">0.6419 +/- 0.0081</td><td align="right">0.0249 +/- 0.0007</td>
      <td align="right">0.8240 +/- 0.0035</td><td align="right">0.8227 +/- 0.0018</td><td align="right">0.7740 +/- 0.0033</td><td align="right">0.0197 +/- 0.0007</td>
      <td align="right">0.7276 +/- 0.0145</td><td align="right">0.7074 +/- 0.0083</td><td align="right">0.6464 +/- 0.0113</td><td align="right">0.0269 +/- 0.0054</td>
      <td align="right">0.7090 +/- 0.0129</td><td align="right">0.7277 +/- 0.0009</td><td align="right">0.6514 +/- 0.0114</td><td align="right">0.0372 +/- 0.0026</td>
      <td align="right">0.7196 +/- 0.0019</td><td align="right">0.7709 +/- 0.0018</td><td align="right">0.6400 +/- 0.0080</td><td align="right">0.0389 +/- 0.0018</td>
      <td align="right">0.6037 +/- 0.0012</td><td align="right">0.5934 +/- 0.0056</td><td align="right">0.4905 +/- 0.0047</td><td align="right">0.0343 +/- 0.0046</td>
      <td align="right">0.7568 +/- 0.0034</td><td align="right">0.7417 +/- 0.0020</td><td align="right">0.6594 +/- 0.0045</td><td align="right">0.0274 +/- 0.0019</td>
    </tr>
    <tr>
      <td><code>LLM-Select</code></td>
      <td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td>
    </tr>
    <tr>
      <td><code>LLM-Lasso</code></td>
      <td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td>
    </tr>
    <tr>
      <td><code>LogisticRegression</code></td>
      <td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td>
    </tr>
    <tr>
      <td><code>XGBoost</code></td>
      <td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td>
    </tr>
    <tr>
      <td><code>CatBoost</code></td>
      <td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td>
    </tr>
    <tr>
      <td><code>RandomForest</code></td>
      <td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td>
    </tr>
    <tr>
      <td><code>SVM</code></td>
      <td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td>
    </tr>
    <tr>
      <td><code>TabPFN</code></td>
      <td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td>
    </tr>
  </tbody>
</table>

### F1 Metrics

<table>
  <thead>
    <tr>
      <th rowspan="2">method</th>
      <th colspan="4"><code>acsincome</code></th>
      <th colspan="4"><code>acsemployment</code></th>
      <th colspan="4"><code>acsemploymentfiltered</code></th>
      <th colspan="4"><code>acshealthinsurance</code></th>
      <th colspan="4"><code>acsincomepovertyratio</code></th>
      <th colspan="4"><code>acsmobility</code></th>
      <th colspan="4"><code>acspubliccoverage</code></th>
      <th colspan="4"><code>acstraveltime</code></th>
      <th colspan="4"><code>ACS AVG</code></th>
    </tr>
    <tr>
      <th>ID</th><th>OOD MEAN</th><th>OOD WORST</th><th>OOD STD</th>
      <th>ID</th><th>OOD MEAN</th><th>OOD WORST</th><th>OOD STD</th>
      <th>ID</th><th>OOD MEAN</th><th>OOD WORST</th><th>OOD STD</th>
      <th>ID</th><th>OOD MEAN</th><th>OOD WORST</th><th>OOD STD</th>
      <th>ID</th><th>OOD MEAN</th><th>OOD WORST</th><th>OOD STD</th>
      <th>ID</th><th>OOD MEAN</th><th>OOD WORST</th><th>OOD STD</th>
      <th>ID</th><th>OOD MEAN</th><th>OOD WORST</th><th>OOD STD</th>
      <th>ID</th><th>OOD MEAN</th><th>OOD WORST</th><th>OOD STD</th>
      <th>ID</th><th>OOD MEAN</th><th>OOD WORST</th><th>OOD STD</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>MLP CrossEntropyLoss</code></td>
      <td align="right">0.4759 +/- 0.0053</td><td align="right">0.6332 +/- 0.0135</td><td align="right">0.5540 +/- 0.0157</td><td align="right">0.0405 +/- 0.0016</td>
      <td align="right">0.8392 +/- 0.0048</td><td align="right">0.7994 +/- 0.0017</td><td align="right">0.6460 +/- 0.0090</td><td align="right">0.0322 +/- 0.0012</td>
      <td align="right">0.8349 +/- 0.0016</td><td align="right">0.7996 +/- 0.0010</td><td align="right">0.6450 +/- 0.0059</td><td align="right">0.0324 +/- 0.0011</td>
      <td align="right">0.4233 +/- 0.0256</td><td align="right">0.2893 +/- 0.0180</td><td align="right">0.1158 +/- 0.0087</td><td align="right">0.0538 +/- 0.0031</td>
      <td align="right">0.4102 +/- 0.2637</td><td align="right">0.4499 +/- 0.2965</td><td align="right">0.3744 +/- 0.2510</td><td align="right">0.0344 +/- 0.0196</td>
      <td align="right">0.8345 +/- 0.0022</td><td align="right">0.8382 +/- 0.0014</td><td align="right">0.7910 +/- 0.0026</td><td align="right">0.0244 +/- 0.0004</td>
      <td align="right">0.5261 +/- 0.0073</td><td align="right">0.5649 +/- 0.0052</td><td align="right">0.4812 +/- 0.0243</td><td align="right">0.0363 +/- 0.0017</td>
      <td align="right">0.3695 +/- 0.0501</td><td align="right">0.3498 +/- 0.0549</td><td align="right">0.2341 +/- 0.0758</td><td align="right">0.0423 +/- 0.0045</td>
      <td align="right">0.5892 +/- 0.0381</td><td align="right">0.5905 +/- 0.0402</td><td align="right">0.4802 +/- 0.0356</td><td align="right">0.0371 +/- 0.0028</td>
    </tr>
    <tr>
      <td><code>FeatureImportanceTargetCELoss</code></td>
      <td align="right">0.4739 +/- 0.0093</td><td align="right">0.6057 +/- 0.0116</td><td align="right">0.5305 +/- 0.0156</td><td align="right">0.0402 +/- 0.0022</td>
      <td align="right">0.8399 +/- 0.0016</td><td align="right">0.7994 +/- 0.0017</td><td align="right">0.6422 +/- 0.0121</td><td align="right">0.0323 +/- 0.0013</td>
      <td align="right">0.8364 +/- 0.0056</td><td align="right">0.7992 +/- 0.0030</td><td align="right">0.6411 +/- 0.0122</td><td align="right">0.0330 +/- 0.0014</td>
      <td align="right">0.4696 +/- 0.0180</td><td align="right">0.3152 +/- 0.0108</td><td align="right">0.1254 +/- 0.0074</td><td align="right">0.0622 +/- 0.0026</td>
      <td align="right">0.5219 +/- 0.0156</td><td align="right">0.5581 +/- 0.0131</td><td align="right">0.4756 +/- 0.0113</td><td align="right">0.0417 +/- 0.0027</td>
      <td align="right">0.8467 +/- 0.0088</td><td align="right">0.8260 +/- 0.0074</td><td align="right">0.7679 +/- 0.0204</td><td align="right">0.0250 +/- 0.0010</td>
      <td align="right">0.5376 +/- 0.0075</td><td align="right">0.5781 +/- 0.0039</td><td align="right">0.4979 +/- 0.0049</td><td align="right">0.0362 +/- 0.0012</td>
      <td align="right">0.4254 +/- 0.0519</td><td align="right">0.3963 +/- 0.0509</td><td align="right">0.2832 +/- 0.0371</td><td align="right">0.0491 +/- 0.0125</td>
      <td align="right">0.6189 +/- 0.0077</td><td align="right">0.6098 +/- 0.0043</td><td align="right">0.4955 +/- 0.0022</td><td align="right">0.0400 +/- 0.0017</td>
    </tr>
    <tr>
      <td><code>LLM Attribution Regularizer</code></td>
      <td align="right">0.4479 +/- 0.0336</td><td align="right">0.6151 +/- 0.0358</td><td align="right">0.5340 +/- 0.0391</td><td align="right">0.0416 +/- 0.0041</td>
      <td align="right">0.8408 +/- 0.0030</td><td align="right">0.8003 +/- 0.0017</td><td align="right">0.6472 +/- 0.0048</td><td align="right">0.0320 +/- 0.0007</td>
      <td align="right">0.8378 +/- 0.0024</td><td align="right">0.8007 +/- 0.0007</td><td align="right">0.6390 +/- 0.0034</td><td align="right">0.0334 +/- 0.0005</td>
      <td align="right">0.4362 +/- 0.0276</td><td align="right">0.3065 +/- 0.0166</td><td align="right">0.1247 +/- 0.0069</td><td align="right">0.0568 +/- 0.0029</td>
      <td align="right">0.5064 +/- 0.0151</td><td align="right">0.5601 +/- 0.0199</td><td align="right">0.4745 +/- 0.0268</td><td align="right">0.0434 +/- 0.0021</td>
      <td align="right">0.8283 +/- 0.0049</td><td align="right">0.8408 +/- 0.0032</td><td align="right">0.7872 +/- 0.0036</td><td align="right">0.0249 +/- 0.0008</td>
      <td align="right">0.5317 +/- 0.0100</td><td align="right">0.5679 +/- 0.0041</td><td align="right">0.4901 +/- 0.0071</td><td align="right">0.0372 +/- 0.0017</td>
      <td align="right">0.4338 +/- 0.0224</td><td align="right">0.4086 +/- 0.0232</td><td align="right">0.2846 +/- 0.0286</td><td align="right">0.0473 +/- 0.0034</td>
      <td align="right">0.6079 +/- 0.0103</td><td align="right">0.6125 +/- 0.0101</td><td align="right">0.4977 +/- 0.0111</td><td align="right">0.0396 +/- 0.0010</td>
    </tr>
    <tr>
      <td><code>LLM-Select</code></td>
      <td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td>
    </tr>
    <tr>
      <td><code>LLM-Lasso</code></td>
      <td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td>
    </tr>
    <tr>
      <td><code>LogisticRegression</code></td>
      <td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td>
    </tr>
    <tr>
      <td><code>XGBoost</code></td>
      <td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td>
    </tr>
    <tr>
      <td><code>CatBoost</code></td>
      <td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td>
    </tr>
    <tr>
      <td><code>RandomForest</code></td>
      <td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td>
    </tr>
    <tr>
      <td><code>SVM</code></td>
      <td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td>
    </tr>
    <tr>
      <td><code>TabPFN</code></td>
      <td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td><td align="center" colspan="4">--</td>
    </tr>
  </tbody>
</table>

### Benchmark Settings Template

Shared ACS protocol:

- Dataset metadata, train/validation state, test states, and removed feature indices
  come from `acs_tasks/config.py`.
- Each method uses the same train/validation split, held-out test states, raw tensor
  cache, and dataset-specific feature removal.
- Class/sample reweighting is selected once per ACS task with the MLP
  CrossEntropyLoss baseline, then reused by every method that supports
  reweighting.
- The selected model or hyperparameter setting is chosen only by validation accuracy
  on the train/validation state; OOD test states are not used for model selection.
- `ID` is the validation metric from the train/validation state. `OOD MEAN`,
  `OOD WORST`, and `OOD STD` are computed across held-out test states.
- Current neural baselines use seeds `[9803, 38224, 8113, 4854, 98825]`.

| method | runner | model/package | selection rule | notes |
|---|---|---|---|---|
| `MLP CrossEntropyLoss` | `run_acs_mlp_ce.py` | PyTorch MLP | best validation accuracy across epochs | Runner exists. Fixed baseline, not broadly optimized. Current setting: `LR=1e-4`, `PATIENCE=500`, `REPEAT=5`, `MAX_EPOCHS=5000`, `TRAIN_BATCH=256`, `EVAL_BATCH=2048`, dataset-specific CE reweighting in the runner. If optimizing, search `LR`, `HIDDEN_SIZE`, and training budget. |
| `FeatureImportanceTargetCELoss` | `run_acs_fitce.py` | PyTorch MLP | best validation accuracy across epochs | Optimized ACS setting. Use `RANKING_METHOD=score_all`, `FEATURE_WEIGHT_MODE=score`, `SUPPRESS_BOUND=0.0`, `LOSS_LAMBDA=16.0`, `LOSS_ALPHA=0.75` (`grad_scale=12.0`, `weight_scale=4.0`), `suppress_scale=1.0`, `TARGET_POWER=1.0`, `REG_SCALE=2.0`, `importance_scale="train_std"`, `REPEAT=5`. |
| `LLM Attribution Regularizer` | `run_acs_laat.py` | PyTorch MLP | best validation accuracy across epochs | Optimized ACS setting. Use `RANKING_METHOD=score`, `FEATURE_WEIGHT_MODE=score`, `REG_SCALE=100`, `LR=1e-4`, `PATIENCE=500`, `REPEAT=5`, `MAX_EPOCHS=5000`, `TRAIN_BATCH=256`, `EVAL_BATCH=2048`, dataset-specific CE reweighting in the runner; no `importance_scale="train_std"` scaling. |
| `LLM-Select` | `run_acs_llm_select.py` | PyTorch MLP | validation accuracy selects feature subset | Runner exists, not optimized yet. Current default: `RANKING_METHOD=score_all`, `SELECTION_MODE=top_p`, `TOP_P_GRID=1.0,0.75,0.5,0.25`; `SCORE_THRESHOLD_GRID=0.0,0.25,0.5,0.75` is only valid for `score` / `score_all`. Search ranking method plus top-p grid; for score methods also search score threshold. |
| `LLM-Lasso` | `run_acs_llm_lasso.py` | scikit-learn L1 logistic regression | validation accuracy selects `eta` and `C` | Runner exists, not optimized yet. Score-based ranking only. Current default: `RANKING_METHOD=score_all`, `ETA_GRID=0,1,2,3,4`, `C_GRID=0.01,0.1,1,10,100`, `PENALTY_FLOOR=0.1`, `CLASS_WEIGHT=none`, train-only `StandardScaler`; `eta=0` recovers plain L1 logistic. Search ranking method, `ETA_GRID`, `C_GRID`, `PENALTY_FLOOR`, and `CLASS_WEIGHT`. |
| `LogisticRegression` | `run_acs_linear.py` | scikit-learn `LogisticRegression` | validation accuracy selects `C` | Runner exists, not broadly optimized. Current default: train-only `StandardScaler`, `C_GRID=0.01,0.1,1,10,100`, `max_iter=1000`, L2 `lbfgs`; runner uses its current dataset-specific `class_weight` policy. If optimizing, search `C_GRID`, `class_weight`, solver/penalty variants, and `max_iter`. |
| `XGBoost` | planned | `xgboost` | validation accuracy selects hyperparameters | use the shared ACS split/cache protocol |
| `CatBoost` | planned | `catboost` | validation accuracy selects hyperparameters | use the shared ACS split/cache protocol |
| `RandomForest` | planned | scikit-learn `RandomForestClassifier` | validation accuracy selects hyperparameters | use the shared ACS split/cache protocol |
| `SVM` | planned | scikit-learn SVM | validation accuracy selects hyperparameters | use the shared ACS split/cache protocol |
| `TabPFN` | planned | `tabpfn` | validation accuracy selects supported settings | use the shared ACS split/cache protocol |

### GPT Ranking Token Cost

Token counts are recorded from `gpt-5.5` ranking JSON files under
`acs_tasks/<dataset>/rankings/`. Each cell is:
`total tokens (input/output tokens, calls)`.

| dataset | rank | score | score_all | seq |
|---|---:|---:|---:|---:|
| `acsincome` | 1211 (608/603, 1 call) | 38332 (28160/10172, 50 calls) | 6899 (3330/3569, 5 calls) | 8208 (5949/2259, 10 calls) |
| `acsemployment` | 1684 (634/1050, 1 call) | 66789 (45010/21779, 80 calls) | 8738 (3460/5278, 5 calls) | 16828 (9935/6893, 16 calls) |
| `acsemploymentfiltered` | 1840 (640/1200, 1 call) | 72909 (47910/24999, 85 calls) | 9287 (3490/5797, 5 calls) | 18202 (10658/7544, 17 calls) |
| `acshealthinsurance` | 2194 (685/1509, 1 call) | 118524 (70405/48119, 125 calls) | 11115 (3715/7400, 5 calls) | 28169 (16799/11370, 25 calls) |
| `acsincomepovertyratio` | 1746 (658/1088, 1 call) | 85842 (56580/29262, 100 calls) | 9935 (3580/6355, 5 calls) | 20894 (12899/7995, 20 calls) |
| `acsmobility` | 2010 (661/1349, 1 call) | 101583 (59200/42383, 105 calls) | 10999 (3595/7404, 5 calls) | 25055 (13607/11448, 21 calls) |
| `acspubliccoverage` | 1902 (648/1254, 1 call) | 79943 (53530/26413, 95 calls) | 9816 (3530/6286, 5 calls) | 19849 (12064/7785, 19 calls) |
| `acstraveltime` | 1880 (639/1241, 1 call) | 68943 (45200/23743, 80 calls) | 8998 (3485/5513, 5 calls) | 11608 (4543/7065, 16 calls) |

## Appendix

### Environment Setup
```bash
conda create -n ood python=3.10.19
conda activate ood
pip install -r requirements.txt
```
