# Dividing vs. Not Dividing lrparam by nbatches

SGD, ADAGRAD, ADADELTA: in SGD, ADAGRAD and ADADELTA, lrparam is NOT divided by batchsize at each iteration.

![AIFB Not Rescaled](aifb_NR_200.png)


SGD, ADAGRAD, ADADELTA: in SGD, ADAGRAD and ADADELTA, lrparam is divided by batchsize at each iteration.

![AIFB Rescaled](aifb_R_200.png)


SGD, ADAGRAD, ADADELTA: in SGD, ADAGRAD and ADADELTA, lrparam is divided by nbatches at each iteration.

![AIFB Rescaled](aifb_H_200.png)
