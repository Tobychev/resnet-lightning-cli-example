# resnet-lightning-cli-example
Project to implement the DAWNnet of the [how to train your resnet example](https://myrtle.ai/learn/how-to-train-your-resnet-1-baseline/) using pytorch lightning with cli support

Right now it's not working right, I include below the performance numbers achived using the myrtle.ai code on my test hardware.
```
 epoch           lr   train time   train loss    train acc   valid time   valid loss    valid acc   total time
 1       0.0067      32.8558       0.0228       0.9937       2.0475       0.2013       0.9415      32.8558
 2       0.0133      33.2363       0.0312       0.9904       1.9934       0.2356       0.9297      33.2363
 3       0.0200      33.4084       0.0665       0.9766       2.0513       0.3201       0.9080      33.4084
 4       0.0267      33.5253       0.1112       0.9613       2.0038       0.4656       0.8645      33.5253
 5       0.0333      33.5853       0.1539       0.9457       2.0096       0.4146       0.8744      33.5853
 6       0.0400      33.6507       0.1898       0.9338       2.0482       0.4272       0.8662      33.6507
 7       0.0467      33.7365       0.2102       0.9253       2.0348       0.3523       0.8817      33.7365
 8       0.0533      33.7512       0.2351       0.9180       2.0308       0.5831       0.8189      33.7512
 9       0.0600      33.7144       0.2453       0.9148       2.0448       0.4487       0.8499      33.7144
10       0.0667      33.7843       0.2600       0.9096       2.0988       0.4415       0.8459      33.7843
11       0.0733      33.8047       0.2665       0.9079       2.0851       0.4232       0.8617      33.8047
12       0.0800      33.7964       0.2761       0.9049       2.0349       0.5210       0.8262      33.7964
13       0.0867      33.7636       0.2848       0.9014       2.0522       0.5718       0.8214      33.7636
14       0.0933      33.7530       0.2970       0.8971       2.0294       0.5175       0.8242      33.7530
15       0.1000      33.8115       0.3072       0.8954       2.1033       0.6456       0.7948      33.8115
16       0.0937      33.8042       0.3032       0.8945       2.1119       0.4889       0.8392      33.8042
17       0.0873      33.8180       0.2859       0.9021       2.0345       0.4146       0.8668      33.8180
18       0.0810      33.7879       0.2686       0.9072       2.0408       0.3683       0.8767      33.7879
19       0.0747      33.7481       0.2563       0.9121       2.0176       0.3905       0.8718      33.7481
20       0.0683      33.7369       0.2361       0.9195       2.0617       0.3739       0.8805      33.7369
```
