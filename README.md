## LOW-BIT-EVA

low-bit-eva/kfac/eva.py : 原32bit eva

```
Model: res20 Optimizer: eva Batch Size: 1, Forward: 8.8092ms, Backward: 13.0430ms, Optimize: 11.5919ms, Step: 33.5601ms

Model: autoencoder Optimizer: eva Batch Size: 1, Forward: 2.2436ms, Backward: 3.7885ms, Optimize: 4.8511ms, Step: 11.0076ms

Model: vit Optimizer: eva Batch Size: 1, Forward: 9.1456ms, Backward: 19.4285ms, Optimize: 15.5988ms, Step: 44.2956ms
Wrote profile results to test_time.py.lprof
Timer unit: 1e-06 s

Total time: 17.6417 s
File: /workspace/low-bit-eva/model/resnet20.py
Function: forward at line 65

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    65                                               @profile
    66                                               def forward(self, x):
    67      2020    1500570.5    742.9      8.5          x = self.pool(F.relu(self.norm1(self.conv1(x))))
    68                                                   
    69      2020       1026.1      0.5      0.0          identity = x
    70      2020     933431.5    462.1      5.3          x = F.relu(self.norm16_1(self.conv16_1(x)))
    71      2020     835356.5    413.5      4.7          x = self.norm16_2(self.conv16_2(x))
    72      2020      30741.8     15.2      0.2          x += identity
    73      2020      40161.0     19.9      0.2          x = F.relu(x)
    74                                                   
    75      2020        821.7      0.4      0.0          identity = x
    76      2020     905988.8    448.5      5.1          x = F.relu(self.norm16_3(self.conv16_3(x)))
    77      2020     885522.4    438.4      5.0          x = self.norm16_4(self.conv16_4(x))
    78      2020      28613.8     14.2      0.2          x += identity
    79      2020      37913.0     18.8      0.2          x = F.relu(x)
    80                                                   
    81      2020      69975.8     34.6      0.4          identity = F.pad(x,(0,0,0,0,0,16))
    82      2020     826450.9    409.1      4.7          x = F.relu(self.norm16_5(self.conv16_5(x)))
    83      2020     788953.8    390.6      4.5          x = self.norm16_6(self.conv16_6(x))
    84      2020      27988.0     13.9      0.2          x += identity
    85      2020      38548.7     19.1      0.2          x = F.relu(x)
    86                                                   
    87                                           
    88      2020      30913.9     15.3      0.2          identity = F.avg_pool2d(identity,1,2)
    89      2020     898138.0    444.6      5.1          x = F.relu(self.norm32_1(self.conv32_1(x)))
    90      2020     814010.4    403.0      4.6          x = self.norm32_2(self.conv32_2(x))
    91      2020      27702.3     13.7      0.2          x += identity
    92      2020      38091.2     18.9      0.2          x = F.relu(x)
    93                                                   
    94      2020       2417.9      1.2      0.0          identity = x
    95      2020     834381.7    413.1      4.7          x = F.relu(self.norm32_3(self.conv32_3(x)))
    96      2020     782763.5    387.5      4.4          x = self.norm32_4(self.conv32_4(x))
    97      2020      27622.7     13.7      0.2          x += identity
    98      2020      37461.5     18.5      0.2          x = F.relu(x)
    99                                                   
   100      2020      65153.4     32.3      0.4          identity = F.pad(x,(0,0,0,0,0,32))
   101      2020     810073.8    401.0      4.6          x = F.relu(self.norm32_5(self.conv32_5(x)))
   102      2020     772317.1    382.3      4.4          x = self.norm32_6(self.conv32_6(x))
   103      2020      26307.6     13.0      0.1          x += identity
   104      2020      36521.6     18.1      0.2          x = F.relu(x)
   105                                                   
   106                                                   
   107      2020      29591.5     14.6      0.2          identity = F.avg_pool2d(x,1,2)
   108      2020     805207.2    398.6      4.6          x = F.relu(self.norm64_1(self.conv64_1(x)))
   109      2020     756797.9    374.7      4.3          x = self.norm64_2(self.conv64_2(x))
   110      2020      25937.5     12.8      0.1          x += identity
   111      2020      36189.5     17.9      0.2          x = F.relu(x)
   112                                                   
   113      2020       2169.5      1.1      0.0          identity = x
   114      2020     778385.1    385.3      4.4          x = F.relu(self.norm64_3(self.conv64_3(x)))
   115      2020     785777.4    389.0      4.5          x = self.norm64_4(self.conv64_4(x))
   116      2020      25369.4     12.6      0.1          x += identity
   117      2020      35436.3     17.5      0.2          x = F.relu(x)
   118                                                   
   119      2020        709.6      0.4      0.0          identity = x
   120      2020     765751.1    379.1      4.3          x = F.relu(self.norm64_5(self.conv64_5(x)))
   121      2020     779915.3    386.1      4.4          x = self.norm64_6(self.conv64_6(x))
   122      2020      25026.7     12.4      0.1          x += identity
   123      2020      34408.4     17.0      0.2          x = F.relu(x)
   124                                                   
   125                                                   
   126      2020      74722.5     37.0      0.4          x = self.pool2(x)
   127      2020      15423.7      7.6      0.1          x = torch.flatten(x, 1)        
   128      2020     508246.2    251.6      2.9          x = self.fc(x)
   129                                                   
   130      2020        713.6      0.4      0.0          return x
```

low-bit-eva/kfac/eva_8bit.py 原8bit eva

```
Model: res20 Optimizer: eva8bit Batch Size: 1, Forward: 11.9066ms, Backward: 20.7832ms, Optimize: 18.5749ms, Step: 51.3755ms

Model: autoencoder Optimizer: eva8bit Batch Size: 1, Forward: 4.6183ms, Backward: 7.6695ms, Optimize: 8.6217ms, Step: 21.0591ms

Model: vit Optimizer: eva8bit Batch Size: 1, Forward: 12.3795ms, Backward: 24.0628ms, Optimize: 24.2708ms, Step: 60.8251ms
Wrote profile results to test_time.py.lprof
Timer unit: 1e-06 s

Total time: 23.4956 s
File: /workspace/low-bit-eva/model/resnet20.py
Function: forward at line 65

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    65                                               @profile
    66                                               def forward(self, x):
    67      2020    1817416.6    899.7      7.7          x = self.pool(F.relu(self.norm1(self.conv1(x))))
    68                                                   
    69      2020        909.6      0.5      0.0          identity = x
    70      2020    1363925.7    675.2      5.8          x = F.relu(self.norm16_1(self.conv16_1(x)))
    71      2020    1274387.6    630.9      5.4          x = self.norm16_2(self.conv16_2(x))
    72      2020      31059.1     15.4      0.1          x += identity
    73      2020      40243.4     19.9      0.2          x = F.relu(x)
    74                                                   
    75      2020        752.9      0.4      0.0          identity = x
    76      2020    1278599.0    633.0      5.4          x = F.relu(self.norm16_3(self.conv16_3(x)))
    77      2020    1199544.0    593.8      5.1          x = self.norm16_4(self.conv16_4(x))
    78      2020      27833.7     13.8      0.1          x += identity
    79      2020      37478.9     18.6      0.2          x = F.relu(x)
    80                                                   
    81      2020      67924.9     33.6      0.3          identity = F.pad(x,(0,0,0,0,0,16))
    82      2020    1196905.1    592.5      5.1          x = F.relu(self.norm16_5(self.conv16_5(x)))
    83      2020    1127494.9    558.2      4.8          x = self.norm16_6(self.conv16_6(x))
    84      2020      25823.5     12.8      0.1          x += identity
    85      2020      35558.1     17.6      0.2          x = F.relu(x)
    86                                                   
    87                                           
    88      2020      28883.3     14.3      0.1          identity = F.avg_pool2d(identity,1,2)
    89      2020    1175011.4    581.7      5.0          x = F.relu(self.norm32_1(self.conv32_1(x)))
    90      2020    1075812.6    532.6      4.6          x = self.norm32_2(self.conv32_2(x))
    91      2020      24177.5     12.0      0.1          x += identity
    92      2020      33374.4     16.5      0.1          x = F.relu(x)
    93                                                   
    94      2020       2166.5      1.1      0.0          identity = x
    95      2020    1078073.0    533.7      4.6          x = F.relu(self.norm32_3(self.conv32_3(x)))
    96      2020    1039843.3    514.8      4.4          x = self.norm32_4(self.conv32_4(x))
    97      2020      23563.1     11.7      0.1          x += identity
    98      2020      32912.9     16.3      0.1          x = F.relu(x)
    99                                                   
   100      2020      59251.4     29.3      0.3          identity = F.pad(x,(0,0,0,0,0,32))
   101      2020    1068415.2    528.9      4.5          x = F.relu(self.norm32_5(self.conv32_5(x)))
   102      2020    1036843.2    513.3      4.4          x = self.norm32_6(self.conv32_6(x))
   103      2020      23363.3     11.6      0.1          x += identity
   104      2020      32844.8     16.3      0.1          x = F.relu(x)
   105                                                   
   106                                                   
   107      2020      26773.5     13.3      0.1          identity = F.avg_pool2d(x,1,2)
   108      2020    1069958.5    529.7      4.6          x = F.relu(self.norm64_1(self.conv64_1(x)))
   109      2020    1028119.0    509.0      4.4          x = self.norm64_2(self.conv64_2(x))
   110      2020      23272.2     11.5      0.1          x += identity
   111      2020      32627.9     16.2      0.1          x = F.relu(x)
   112                                                   
   113      2020       2206.2      1.1      0.0          identity = x
   114      2020    1053443.7    521.5      4.5          x = F.relu(self.norm64_3(self.conv64_3(x)))
   115      2020    1014343.3    502.2      4.3          x = self.norm64_4(self.conv64_4(x))
   116      2020      23125.0     11.4      0.1          x += identity
   117      2020      32434.6     16.1      0.1          x = F.relu(x)
   118                                                   
   119      2020        663.8      0.3      0.0          identity = x
   120      2020    1047212.3    518.4      4.5          x = F.relu(self.norm64_5(self.conv64_5(x)))
   121      2020    1008528.4    499.3      4.3          x = self.norm64_6(self.conv64_6(x))
   122      2020      22968.0     11.4      0.1          x += identity
   123      2020      32107.6     15.9      0.1          x = F.relu(x)
   124                                                   
   125                                                   
   126      2020      68117.3     33.7      0.3          x = self.pool2(x)
   127      2020      14708.7      7.3      0.1          x = torch.flatten(x, 1)        
   128      2020     733950.2    363.3      3.1          x = self.fc(x)
   129                                                   
   130      2020        656.0      0.3      0.0          return x
```

low-bit-eva/kfac/eva_8bit_cpu.py 在cpu量化a,g

```
Model: res20 Optimizer: eva8bit_cpu Batch Size: 1, Forward: 16.0229ms, Backward: 17.0633ms, Optimize: 22.2904ms, Step: 55.4954ms

Model: autoencoder Optimizer: eva8bit_cpu Batch Size: 1, Forward: 5.4256ms, Backward: 6.0297ms, Optimize: 12.1970ms, Step: 23.8224ms

Model: vit Optimizer: eva8bit_cpu Batch Size: 1, Forward: 18.7381ms, Backward: 23.8324ms, Optimize: 28.7378ms, Step: 71.4238ms
Wrote profile results to test_time.py.lprof
Timer unit: 1e-06 s

Total time: 31.0041 s
File: /workspace/low-bit-eva/model/resnet20.py
Function: forward at line 65

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    65                                               @profile
    66                                               def forward(self, x):
    67      2020    1833237.8    907.5      5.9          x = self.pool(F.relu(self.norm1(self.conv1(x))))
    68                                                   
    69      2020       1060.0      0.5      0.0          identity = x
    70      2020    1360425.1    673.5      4.4          x = F.relu(self.norm16_1(self.conv16_1(x)))
    71      2020    1271291.9    629.4      4.1          x = self.norm16_2(self.conv16_2(x))
    72      2020     111989.1     55.4      0.4          x += identity
    73      2020      84154.3     41.7      0.3          x = F.relu(x)
    74                                                   
    75      2020        859.0      0.4      0.0          identity = x
    76      2020    1448911.5    717.3      4.7          x = F.relu(self.norm16_3(self.conv16_3(x)))
    77      2020    1473462.1    729.4      4.8          x = self.norm16_4(self.conv16_4(x))
    78      2020     108139.8     53.5      0.3          x += identity
    79      2020     119824.1     59.3      0.4          x = F.relu(x)
    80                                                   
    81      2020     117269.5     58.1      0.4          identity = F.pad(x,(0,0,0,0,0,16))
    82      2020    1483463.9    734.4      4.8          x = F.relu(self.norm16_5(self.conv16_5(x)))
    83      2020    1441465.8    713.6      4.6          x = self.norm16_6(self.conv16_6(x))
    84      2020      95577.1     47.3      0.3          x += identity
    85      2020     104704.7     51.8      0.3          x = F.relu(x)
    86                                                   
    87                                           
    88      2020      90381.4     44.7      0.3          identity = F.avg_pool2d(identity,1,2)
    89      2020    1565276.7    774.9      5.0          x = F.relu(self.norm32_1(self.conv32_1(x)))
    90      2020    1402401.2    694.3      4.5          x = self.norm32_2(self.conv32_2(x))
    91      2020     105472.0     52.2      0.3          x += identity
    92      2020     106163.5     52.6      0.3          x = F.relu(x)
    93                                                   
    94      2020      11352.1      5.6      0.0          identity = x
    95      2020    1490830.3    738.0      4.8          x = F.relu(self.norm32_3(self.conv32_3(x)))
    96      2020    1399308.0    692.7      4.5          x = self.norm32_4(self.conv32_4(x))
    97      2020     103270.4     51.1      0.3          x += identity
    98      2020     107435.0     53.2      0.3          x = F.relu(x)
    99                                                   
   100      2020     101872.3     50.4      0.3          identity = F.pad(x,(0,0,0,0,0,32))
   101      2020    1433152.7    709.5      4.6          x = F.relu(self.norm32_5(self.conv32_5(x)))
   102      2020    1405478.7    695.8      4.5          x = self.norm32_6(self.conv32_6(x))
   103      2020     101474.8     50.2      0.3          x += identity
   104      2020     104944.1     52.0      0.3          x = F.relu(x)
   105                                                   
   106                                                   
   107      2020      98952.2     49.0      0.3          identity = F.avg_pool2d(x,1,2)
   108      2020    1447465.0    716.6      4.7          x = F.relu(self.norm64_1(self.conv64_1(x)))
   109      2020    1403824.5    695.0      4.5          x = self.norm64_2(self.conv64_2(x))
   110      2020     103200.8     51.1      0.3          x += identity
   111      2020     103744.6     51.4      0.3          x = F.relu(x)
   112                                                   
   113      2020      11157.9      5.5      0.0          identity = x
   114      2020    1472103.0    728.8      4.7          x = F.relu(self.norm64_3(self.conv64_3(x)))
   115      2020    1383159.9    684.7      4.5          x = self.norm64_4(self.conv64_4(x))
   116      2020     102919.8     51.0      0.3          x += identity
   117      2020     110260.7     54.6      0.4          x = F.relu(x)
   118                                                   
   119      2020        713.6      0.4      0.0          identity = x
   120      2020    1490583.4    737.9      4.8          x = F.relu(self.norm64_5(self.conv64_5(x)))
   121      2020    1402163.7    694.1      4.5          x = self.norm64_6(self.conv64_6(x))
   122      2020     101481.0     50.2      0.3          x += identity
   123      2020     105216.3     52.1      0.3          x = F.relu(x)
   124                                                   
   125                                                   
   126      2020     140405.6     69.5      0.5          x = self.pool2(x)
   127      2020      47526.2     23.5      0.2          x = torch.flatten(x, 1)        
   128      2020     893751.4    442.5      2.9          x = self.fc(x)
   129                                                   
   130      2020        779.7      0.4      0.0          return x
```

