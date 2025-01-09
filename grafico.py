import random
import subprocess
import time
import matplotlib.pyplot as plt
import numpy as np

def generate_random_integer(c, x):
    # Generate a random number from a Gaussian (normal) distribution
    return int(round(random.gauss(c, x)))

def generate_request(requests, facts):
    r = int(random.uniform(1, facts))
    if requests.get(r) is None:
        requests[r] = True
        return r
    else:
        return generate_request(requests, facts)

def generate_input(num_factories, num_countries, num_children, variance, max_cap, max_requests):
    factories_data = {}
    countries_data = {}
    children_data = {}

    avg_fs_per_country = int(num_factories / num_countries)
    avg_cs_per_country = int(num_children / num_countries)

    total_fs = 0
    total_cs = 0

    countries_info = []
    cur_fact = 1
    cur_child = 1

    for c in range(num_countries):
        cur_fs = generate_random_integer(avg_fs_per_country, avg_fs_per_country * variance)
        cur_cs = generate_random_integer(avg_cs_per_country, avg_cs_per_country * variance)

        if (num_factories - total_fs < cur_fs or c == (num_countries - 1)):
            cur_fs = num_factories - total_fs

        if (num_children - total_cs < cur_cs or c == (num_countries - 1)):
            cur_cs = num_children - total_cs

        countries_info.append((cur_fs, cur_cs))
        total_fs += cur_fs
        total_cs += cur_cs

        cur_total_cap = 0
        for j in range(cur_fact, total_fs + 1):
            cap = int(random.uniform(1, max_cap))
            cur_total_cap += cap
            factories_data[j] = (j, c + 1, cap)

        country_export_cap = int(random.uniform(cur_total_cap / 4, cur_total_cap))
        country_min_cs = int(random.uniform(cur_cs / 4, cur_cs))
        countries_data[c + 1] = (c + 1, country_export_cap, country_min_cs)

        for ch in range(cur_child, total_cs + 1):
            requests_num = int(random.uniform(1, max_requests))
            requests = {}
            lst = [ch, c + 1]
            for i in range(requests_num):
                r = generate_request(requests, num_factories)
                lst.append(r)
            children_data[ch] = lst

        cur_fact = total_fs + 1
        cur_child = total_cs + 1

    # Format data for input
    input_data = f"{num_factories} {num_countries} {num_children}\n"
    for i in range(num_factories):
        fi, pj, fmaxi = factories_data[i + 1]
        input_data += f"{fi} {pj} {fmaxi}\n"
    for i in range(num_countries):
        pj, pmaxj, pminj = countries_data[i + 1]
        input_data += f"{pj} {pmaxj} {pminj}\n"
    for i in range(num_children):
        c_data = children_data[i + 1]
        input_data += " ".join(map(str, c_data)) + "\n"

    return input_data

def simulate_execution(input_data):
    # Execute your project and measure time
    start_time = time.time()
    result = subprocess.run(
        ["python3", "projeto3.py"],
        input=input_data,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    end_time = time.time()

    # Measure execution time and return
    execution_time = end_time - start_time
    output = result.stdout.strip()
    try:
        output_value = int(output)
    except ValueError:
        output_value = -1  # Em caso de erro na conversão
    return execution_time, output_value

# Run experiments
results = []
increments = 10  # Increment size for n, m
initial_values = (10, 3)  # Initial values for n and m
num_tests = 100
k = 5  # Proportionality constant for t = k * n

for i in range(num_tests):
    n = initial_values[0] + i * increments
    m = initial_values[1] + i * increments
    t = k * n  # Ensure t grows proportionally to n
    variance = 0.1
    max_cap = 10 + i * 2  # Aumenta em cada teste
    max_requests = 5 + i  # Aumenta linearmente

    # Generate input and retry if output is -1
    output_value = -1
    while output_value == -1:
        input_data = generate_input(n, m, t, variance, max_cap, max_requests)
        exec_time, output_value = simulate_execution(input_data)
    
    print(f"n={n}, m={m}, t={t}, time={exec_time}, test={i}")
    results.append((n, m, t, exec_time))

# Sort and calculate fitting polynomial
results.sort(key=lambda x: x[2])
complexities = [r[2] for r in results]
times = [r[3] for r in results]

# Fit a polynomial to the data
poly_coeffs = np.polyfit(complexities, times, 2)  # Quadratic fit
poly_fn = np.poly1d(poly_coeffs)
sorted_complexities = sorted(complexities)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(sorted_complexities, [poly_fn(x) for x in sorted_complexities], '--', label="Tendência global", color="red")
plt.plot(complexities, times, 'o', label="Dados experimentais", color="blue")

# Configurações do gráfico
plt.xlabel("f(n,m,t) = n x l", fontsize=12)
plt.ylabel("Tempo (s)", fontsize=12)
plt.title("Curva de Tendência para Tempo de Execução em Função da Soma de variáveis", fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Print table
print(f"{'Factories':<10}{'Countries':<10}{'Children':<10}{'Execution Time (s)'}")
for r in results:
    print(f"{r[0]:<10}{r[1]:<10}{r[2]:<10}{r[3]:.4f}")
