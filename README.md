# fashion_mnist

Performance comparison of XgBoost when applied to MNIST and Fashion-MNIST datasets

```
data/fashion
(60000, 784) (60000,)
(10000, 784) (10000,)
[0]     train-merror:0.17704    validation-merror:0.20467
Multiple eval metrics have been passed: 'validation-merror' will be used for early stopping.

Will train until validation-merror hasn't improved in 50 rounds.
[1]     train-merror:0.15683    validation-merror:0.18383
[2]     train-merror:0.15035    validation-merror:0.17783
[3]     train-merror:0.14678    validation-merror:0.17517
[4]     train-merror:0.14422    validation-merror:0.17400
[5]     train-merror:0.14244    validation-merror:0.16983
[6]     train-merror:0.14031    validation-merror:0.17217
[7]     train-merror:0.13913    validation-merror:0.16950
[8]     train-merror:0.13811    validation-merror:0.16967
[9]     train-merror:0.13772    validation-merror:0.16783
[10]    train-merror:0.13593    validation-merror:0.16450
[11]    train-merror:0.13459    validation-merror:0.16567
[12]    train-merror:0.13370    validation-merror:0.16417
[13]    train-merror:0.13218    validation-merror:0.16267
[14]    train-merror:0.13130    validation-merror:0.16133
[15]    train-merror:0.13035    validation-merror:0.16083
[16]    train-merror:0.12980    validation-merror:0.15917
[17]    train-merror:0.12865    validation-merror:0.16017
[18]    train-merror:0.12756    validation-merror:0.15967
[19]    train-merror:0.12661    validation-merror:0.15767
[20]    train-merror:0.12578    validation-merror:0.15683
[450]   train-merror:0.00596    validation-merror:0.09950
[451]   train-merror:0.00594    validation-merror:0.09933
[452]   train-merror:0.00587    validation-merror:0.09933
[453]   train-merror:0.00585    validation-merror:0.09933
[454]   train-merror:0.00583    validation-merror:0.09950
[455]   train-merror:0.00578    validation-merror:0.09933
[456]   train-merror:0.00574    validation-merror:0.09917
[457]   train-merror:0.00572    validation-merror:0.09917
[458]   train-merror:0.00570    validation-merror:0.09950
[459]   train-merror:0.00572    validation-merror:0.09900
[460]   train-merror:0.00565    validation-merror:0.09933
[461]   train-merror:0.00565    validation-merror:0.09933
[462]   train-merror:0.00561    validation-merror:0.09967
[463]   train-merror:0.00565    validation-merror:0.09950
[464]   train-merror:0.00565    validation-merror:0.09950
[465]   train-merror:0.00557    validation-merror:0.09983
[466]   train-merror:0.00550    validation-merror:0.09950
[467]   train-merror:0.00544    validation-merror:0.09967
[468]   train-merror:0.00546    validation-merror:0.09950
Stopping. Best iteration:
[418]   train-merror:0.00713    validation-merror:0.09900

981.094 sec
accuracy 89.77%
```

```
data/mnist
(60000, 784) (60000,)
(10000, 784) (10000,)
[0]     train-merror:0.14822    validation-merror:0.16083
Multiple eval metrics have been passed: 'validation-merror' will be used for early stopping.

Will train until validation-merror hasn't improved in 50 rounds.
[1]     train-merror:0.11746    validation-merror:0.13500
[2]     train-merror:0.09878    validation-merror:0.11567
[3]     train-merror:0.09087    validation-merror:0.10950
[4]     train-merror:0.08780    validation-merror:0.10583
[5]     train-merror:0.08456    validation-merror:0.10067
[6]     train-merror:0.08263    validation-merror:0.09750
[7]     train-merror:0.08007    validation-merror:0.09617
[8]     train-merror:0.07854    validation-merror:0.09617
[9]     train-merror:0.07567    validation-merror:0.09233
[10]    train-merror:0.07437    validation-merror:0.09167
[11]    train-merror:0.07265    validation-merror:0.08783
[12]    train-merror:0.07068    validation-merror:0.08483
[13]    train-merror:0.06922    validation-merror:0.08317
[14]    train-merror:0.06763    validation-merror:0.08167
[15]    train-merror:0.06604    validation-merror:0.08050
[16]    train-merror:0.06459    validation-merror:0.07983
[17]    train-merror:0.06346    validation-merror:0.07867
[18]    train-merror:0.06224    validation-merror:0.07800
[19]    train-merror:0.06144    validation-merror:0.07717
[20]    train-merror:0.06059    validation-merror:0.07700
...
[420]   train-merror:0.00241    validation-merror:0.02417
[421]   train-merror:0.00241    validation-merror:0.02417
[422]   train-merror:0.00239    validation-merror:0.02417
[423]   train-merror:0.00237    validation-merror:0.02417
[424]   train-merror:0.00237    validation-merror:0.02417
[425]   train-merror:0.00239    validation-merror:0.02417
[426]   train-merror:0.00239    validation-merror:0.02417
[427]   train-merror:0.00239    validation-merror:0.02433
[428]   train-merror:0.00239    validation-merror:0.02433
[429]   train-merror:0.00237    validation-merror:0.02433
[430]   train-merror:0.00237    validation-merror:0.02450
[431]   train-merror:0.00235    validation-merror:0.02450
[432]   train-merror:0.00233    validation-merror:0.02450
[433]   train-merror:0.00233    validation-merror:0.02450
Stopping. Best iteration:
[383]   train-merror:0.00256    validation-merror:0.02417

866.339 sec
accuracy 96.84%
```
