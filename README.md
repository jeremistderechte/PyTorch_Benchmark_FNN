<h3 align="center">Results</h3>
<table align="center">
  <tr>
    <th>Name</th>
    <th>Architecture</th>
    <th>Memory Type</th>
    <th>Result</th>
  </tr>
  <tr>
    <td>Nvidia Quadro RTX 3000 6GB</td>
    <td>Turing</td>
    <td>GDDR6</td>
    <td></td>
  </tr>
  <tr>
    <td>Nvidia RTX 3090 24GB</td>
    <td>Ampere</td>
    <td>GDDR6X</td>
    <td></td>
  </tr>
  <tr>
    <td>Nvidia RTX 3060 12GB</td>
    <td>Ampere</td>
    <td>GDDR6</td>
    <td>62.20s</td>
  </tr>
  <tr>
    <td>Nvidia Tesla K40c 12GB</td>
    <td>Kepler</td>
    <td>GDDR5</td>
    <td>170.49s</td>
  </tr>
</table>

```bash
wget https://github.com/jeremistderechte/PyTorch_Benchmark_FNN/releases/download/bench_1.0/benchmark_fnn.tar.gz
tar -xzf benchmark_fnn.tar.gz
# Use correct python bin
python benchmark.py 
