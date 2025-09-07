from ortools.sat.python import cp_model

k = 8
g = k**2 // 4 + 1
expected, expected_lax = k//2 - 1, [(k+3)//4 - 1, k - 1]


def main():
    model = cp_model.CpModel()
    ht_vars = {f"{z},{i},{j}": model.new_bool_var(f"{z},{i},{j}") for z in range(g) for i in range(g) for j in range(i+1, g) if z != i and z != j}

    # k/2-1 = 1 = x[3,1,2] + x[4,1,2] + x[5,1,2]: (perechea (1, 2) are voie sa apara o singura data. nu poate aparea in secv. 1/2)
    for i in range(g):
        for j in range(i+1, g):
            model.add(expected == sum([ht_vars[f"{z},{i},{j}"] for z in range(g) if z != i and z != j]))
            print(f"{expected} == " + " + ".join([f"x[{z+1},{i+1},{j+1}]" for z in range(g) if z != i and z != j]))

    # k/2-1 = 1 = x[4,1,2] + x[4,1,3] + x[4,1,5]: 1 are voie sa apara de k/2-1 ori in secv 4.
    # il las sa apara de cel putin k/2-1 ori (e.g. grupa din care face parte sa fie mai mare). incerc sa minimizez numarul de ori de care poate sa apara.
    for z in range(g):
        for i in range(g):
            if i != z:
                model.add(expected_lax[0] <= sum([ht_vars[f"{z},{min(i, j)},{max(i, j)}"] for j in range(g) if j != z and j != i]))
                model.add(sum([ht_vars[f"{z},{min(i, j)},{max(i, j)}"] for j in range(g) if j != z and j != i]) <= expected_lax[1])
                print(f"{expected_lax[0]} <= " + " + ".join([f"x[{z+1},{min(i, j)+1},{max(i, j)+1}]" for j in range(g) if j != z and j != i]) + f" <= {expected_lax[1]}")

    # x[z,i,j] = T & x[z,j,t] = T => x[z,i,t] = T. daca (i, j) si (j, t) sunt in aceeasi grupa in secventa z, atunci si (i, t) trebuie sa fie impreuna.
    slack_vars = []
    for z in range(g):
        for i in range(g):
            if i != z:
                for j in range(i+1, g):
                    if j != z:
                        for t in range(j+1, g):
                            if t != z:
                                slack_vars.append(model.new_bool_var("s{z},{i},{j}"))
                                model.add(2 != ht_vars[f"{z},{i},{j}"] + ht_vars[f"{z},{j},{t}"] + ht_vars[f"{z},{i},{t}"]).only_enforce_if(slack_vars[-1])
                                print(f"2 != x[{z+1},{i+1},{j+1}] + x[{z+1},{j+1},{t+1}] + x[{z+1},{i+1},{t+1}]")

    model.maximize(sum(slack_vars))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60.0
    status = solver.solve(model)

    if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        print(f"No solution found, {status = }.")
        return

    # print(f"status = {'optimal' if status == cp_model.OPTIMAL else 'feasible'}")
    print(f"status = {'optimal' if status == cp_model.OPTIMAL else 'feasible'}, ok soft constraints = {sum([solver.value(var) for var in slack_vars])} / {len(slack_vars)}")

    graphs = [[[] for i in range(g)] for z in range(g)] # g grafuri, unul pentru fiecare secventa.

    for var_name, var in ht_vars.items():
        if solver.value(var) == 1:
            z, i, j = map(int, var_name.split(','))
            graphs[z][i].append(j)
            graphs[z][j].append(i)
            print(f"x[{z+1},{i+1},{j+1}]", end = " ")
    print()

    def dfs(graph, curr_group, viz, group_content, nod):
        viz[nod] = curr_group
        group_content.append(nod + 1)
        for nn in graph[nod]:
            if viz[nn] == -1:
                dfs(graph, curr_group, viz, group_content, nn)

    print("[")
    for z in range(g):
        print("    [", end = '')
        
        viz = [-1] * g
        curr_group = 0
        for i in range(g):
            if i != z and viz[i] == -1:
                group_content = []
                dfs(graphs[z], curr_group, viz, group_content, i)
                curr_group += 1

                print("[" + ', '.join(map(str, group_content)) + "], ", end = '') # (", " if curr_group < k//2 else '')
        print("]" + ("," if z+1 < g else ''))
    print("]")
    print(f"{k = }, # constraints = {len(model.Proto().constraints)}.")


if __name__ == "__main__":
    main()
