# PyTorch Benchmark

This benchmark involves training a simple ReLU network on a Mandelbrot dataset. However, it does not scale properly on more powerful GPUs because it fails to fully utilize all FPUs on the RTX 3090. As a result, the performance scaling from the RTX 3060 to the RTX 3090 is suboptimal. I plan to develop a more complex benchmark to better test and utilize more powerful hardware.

<h3 align="center">Results</h3>
<table align="center">
  <tr>
    <th>Name</th>
    <th>Architecture</th>
    <th>Memory Type</th>
    <th>Release date</th>
    <th>Result</th>
  </tr>
  <tr>
    <td>Nvidia RTX 3090 24GB</td>
    <td>Ampere</td>
    <td>GDDR6X</td>
    <td>2020</td>
    <td>0min 50s</td>
  </tr>
    <tr>
    <td>AMD Radeon PRO VII 16GB</td>
    <td>GCN 5.1</td>
    <td>HBM2</td>
    <td>2020</td>
    <td>0min 53s</td>
  </tr>
  <tr>
    <td>Nvidia RTX 3060 12GB</td>
    <td>Ampere</td>
    <td>GDDR6</td>
    <td>2021</td>
    <td>1min 2s</td>
  </tr>
  <tr>
    <td>Nvidia Tesla T4 16GB (Google Colab)</td>
    <td>Turing</td>
    <td>GDDR6</td>
    <td>2018</td>
    <td>1min 41s</td>
  </tr>
    <tr>
    <td>Nvidia Quadro RTX 3000 6GB</td>
    <td>Turing</td>
    <td>GDDR6</td>
    <td>2019</td>
    <td>1min 49s</td>
  </tr>
  <tr>
    <td>Nvidia Tesla K40c 12GB</td>
    <td>Kepler</td>
    <td>GDDR5</td>
    <td>2013</td>
    <td>2min 50s</td>
  </tr>
  <tr>
    <td>AMD Ryzen 5 3600</td>
    <td>Zen 2</td>
    <td>DDR4</td>
    <td>2019</td>
    <td>20min 37s</td>
  </tr>
  <tr>
    <td>Intel Core Ultra 7 155H</td>
    <td>Meteor Lake</td>
    <td>DDR5</td>
    <td>2023</td>
    <td>25min 50s</td>
  </tr>
</table>
<br>

Prerequisites
- PyTorch
- pandas
- scikit-learn
  

```bash
wget https://github.com/jeremistderechte/PyTorch_Benchmark_FNN/releases/download/bench_1.0/benchmark_fnn.tar.gz
tar -xzf benchmark_fnn.tar.gz
# Use correct python bin
python benchmark.py 
