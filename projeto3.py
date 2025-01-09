from pulp import *
from collections import defaultdict

def validate_input(factories_data, countries_data, children_data):
    # Verificar se existem fábricas com stock
    valid_factories = {f_id for f_id, stock in factories_data.items() if stock > 0}
    if not valid_factories:
        return False
    
    # Calcular totais para verificação rápida
    total_stock = sum(factories_data.values())
    total_min_toys = sum(country['min_toys'] for country in countries_data.values())
    
    # Verificar se há stock suficiente para requisitos mínimos
    if total_stock < total_min_toys:
        return False
        
    # Verificar se algum país tem requisitos impossíveis
    country_children = defaultdict(int)
    for child_id, country_id in children_data['country'].items():
        country_children[country_id] += 1
    
    for country_id, data in countries_data.items():
        if data['min_toys'] > country_children[country_id]:
            return False
    
    return True

def solve_christmas_distribution(input_lines):
    # Parse input de forma mais eficiente
    lines = iter(input_lines)
    n, m, t = map(int, next(lines).split())
    
    # Usar dicionários para acesso O(1)
    factories = {}
    factory_country = {}
    for _ in range(n):
        f_id, c_id, max_stock = map(int, next(lines).split())
        factories[f_id] = max_stock
        factory_country[f_id] = c_id
    
    countries = {}
    for _ in range(m):
        c_id, max_export, min_toys = map(int, next(lines).split())
        countries[c_id] = {'max_export': max_export, 'min_toys': min_toys}
    
    # Estrutura de dados otimizada para crianças
    children = {
        'toys': defaultdict(set),
        'country': {}
    }
    
    for _ in range(t):
        request = list(map(int, next(lines).split()))
        child_id, country_id, *wanted_toys = request
        if wanted_toys:
            children['toys'][child_id] = set(wanted_toys)
            children['country'][child_id] = country_id
    
    # Validação inicial
    if not validate_input(factories, countries, children):
        return -1
    
    # Criar modelo de otimização
    prob = LpProblem("Christmas_Toys_Distribution", LpMaximize)
    
    # Criar variáveis apenas para combinações válidas
    x = {}
    for child_id, wanted_toys in children['toys'].items():
        valid_toys = {f for f in wanted_toys if factories[f] > 0}
        for factory in valid_toys:
            x[child_id, factory] = LpVariable(f"toy_{child_id}_{factory}", cat='Binary')
    
    # Função objetivo otimizada
    prob += lpSum(x.values())
    
    # Restrições otimizadas
    # 1. Uma criança, um presente
    for child_id in children['toys']:
        prob += lpSum(x[child_id, f] for f in children['toys'][child_id] 
                     if (child_id, f) in x) <= 1
    
    # 2. Capacidade das fábricas
    factory_vars = defaultdict(list)
    for (child_id, factory_id) in x:
        factory_vars[factory_id].append(x[child_id, factory_id])
    
    for factory_id, vars_list in factory_vars.items():
        prob += lpSum(vars_list) <= factories[factory_id]
    
    # 3. Limites de exportação por país
    country_exports = defaultdict(list)
    for (child_id, factory_id), var in x.items():
        if factory_country[factory_id] != children['country'][child_id]:
            country_exports[factory_country[factory_id]].append(var)
    
    for country_id, vars_list in country_exports.items():
        prob += lpSum(vars_list) <= countries[country_id]['max_export']
    
    # 4. Mínimo de brinquedos por país
    country_toys = defaultdict(list)
    for (child_id, factory_id), var in x.items():
        country_toys[children['country'][child_id]].append(var)
    
    for country_id, vars_list in country_toys.items():
        prob += lpSum(vars_list) >= countries[country_id]['min_toys']
    
    # Resolver com configurações otimizadas
    prob.solve(GLPK(msg=0, options=["--dual"]))
    return int(pulp.value(prob.objective)) if prob.status == 1 else -1

def main():
    try:
        input_lines = []
        while True:
            line = input()
            input_lines.append(line)
    except EOFError:
        pass
    
    print(solve_christmas_distribution(input_lines))

if __name__ == "__main__":
    main()