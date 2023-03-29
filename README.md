 # [1次元畳み込み処理による音響信号の異常検知]
・回転機器(Bearing)の音響(振動)情報から異常を検知する識別機を作成した。<br>
・ネットワークモデルにはSVM(scikit-learnのlinear)を使用。<br>

■使用データ
・https://dcase.community/challenge2022/task-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring<br>
・上記のDcaseのデータを用いた。今回はDeveloping datasetの中にあるbearingからsection00のtestデータをtrainデータとして異常の音響データ100個、正常の音響データ100個、計200個のデータを使った学習を行った。<br>
　section00には12種類の回転速度が異なるものをデータして格納されている。<br>
・前処理として音響信号を短時間フーリエ変換 → 絶対値を算出 → 対数スケールに変換 → 一次元の手順で実験を行った。
