[1次元畳み込み処理による音響信号の異常検知]
・回転機器(Bearing)の音響(振動)情報から異常を検知する識別機を作成した。
・ネットワークモデルには一次元畳み込みを使用。

■使用データ
・https://dcase.community/challenge2022/task-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring
・上記のDcaseのデータを用いた。今回はDeveloping datasetの中にあるbearingからsection00のtestデータをtrainデータとして異常の音響データ100個、正常の音響データ100個、計200個のデータを使った学習を行った。
　section00には12種類の回転速度が異なるものをデータして格納されている。
・音響信号を短時間フーリエ変換をした出力に絶対値を取った値をネットワークの入力とした。(虚数が含まれるため、絶対値を取った)
