import numpy as np
import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt
import gurobipy as gu
import networkx as nx
import csv
import pickle
import sys, string, os
import pyomo.environ as pe
import pyomo.opt
from scipy.optimize import minimize
import re
import operator

global crea_datos
global calibra

def generate_new_distribution_parameters(dist, flag):
    
    new_dist = {}
    
    
    if flag == 'Norm':
        
        new_dist = dist
    
    elif flag == 'LogNorm':
        
        for key in dist:
                
            mean = dist[key][0]
            std = dist[key][1]
            
            mean_log = np.log(np.power(mean, 2) / np.sqrt(np.power(mean, 2) + np.power(std, 2)))
            std_log = np.sqrt(np.log( 1 + (np.power(std, 2)/np.power(mean, 2))))
            
            new_dist[key] = (mean_log, std_log)
        
    elif flag == 'Triang':
        
        for key in dist:
                
            mean = dist[key][0]
            std = dist[key][1]
            
            def objective_triangular(params):
                a, b, c = params
                mean_tri = (a + b + c) / 3
                std_dev_tri = np.sqrt((a**2 + b**2 + c**2 - a*b - a*c - b*c) / 18)
                return (std_dev_tri - std)**2 + (mean_tri - mean)**2
            
            # Initial guess for parameters (can be adjusted based on your knowledge)
            initial_guess = [mean - std, mean + std, mean]
            
            # Optimization to find parameters that match mean and standard deviation
            result = minimize(objective_triangular, initial_guess, method='Nelder-Mead')
            
            # Extract the optimal parameters
            a_tri, b_tri, c_tri, = result.x
            
            new_dist[key] = (a_tri, b_tri, c_tri)
        
        
    elif flag == 'Uniform':
        
        for key in dist:
                
            mean = dist[key][0]
            std = dist[key][1]
            
            def objective_uniform(params):
                a, b = params
                mean_uni = (a + b) / 2
                std_dev_uni = (b - a) / np.sqrt(12)
                return (std_dev_uni - std)**2 + (mean_uni - mean)**2
            
            # Initial guess for parameters (can be adjusted based on your knowledge)
            initial_guess = [mean - std / 2, mean + std / 2]
            
            # Optimization to find parameters that match mean and standard deviation
            result = minimize(objective_uniform, initial_guess, method='Nelder-Mead')
            
            # Extract the optimal parameters
            a_uni, b_uni = result.x
            
            new_dist[key] = (a_uni, b_uni)
                
        
    return new_dist
    

class Set_Partitioning: 
    def __init__(self, param_nodes, solucionTPP, t, c, Q, daily_time, solver_name):
            
        self.giant = param_nodes #giant-tour 
        self.solucionTPP = solucionTPP
        self.t = t
        self.demand = {i:self.solucionTPP[self.t][1][i] for i in self.giant if i != 0} 
        self.cij = c #unit time between (i,j)
        self.load_capacity = Q #veh capacity
        self.duration = daily_time #max working day
        self.flag_saving = False
        self.solver = solver_name
        
        
        #set partitioning
        self.pool = {}
        self.c_p = {}
        self.a_ij = {}
        
    def Run_SP_TwoOPT(self):
        
        self.add_routes_following_order()
        self.TwoOpt_changes()
        solucion = self.solve_set_partitioning()
        FO, Routes, solutionTPP_ = self.extra_routes(solucion)
        
        return FO, Routes, solutionTPP_
        
    def extra_routes(self, results2):
        
        Rutas_finales = {}
        Rutas_finales[self.t] = {}
        
        index_list = results2.index.to_list()
        
        FO_total = 0
        
        for i in range(len(results2)):
            
            Rutas_finales[self.t][i] = self.pool[results2.loc[index_list[i]]['route']]
            
            FO_total += self.pool[results2.loc[index_list[i]]['route']][2]
            
            for k in self.pool[results2.loc[index_list[i]]['route']][3]:
                self.solucionTPP[self.t][4][k]=i
                self.solucionTPP[self.t][5][k] = self.pool[results2.loc[index_list[i]]['route']][3][k]
                       
        return FO_total, Rutas_finales, self.solucionTPP
                
          
    def solve_set_partitioning(self):
        
        solver = pyomo.opt.SolverFactory(self.solver)
        
        '''
        if options is not None:
            for key, value in options.items():
                solver.options[key] = value
        '''
        
        names = list(self.pool.keys())
        
        model = pe.ConcreteModel()
        
        model.V = pe.Set(initialize=self.demand.keys()) #set of vertices /{0}
        model.R = pe.Set(initialize=range(len(self.pool))) #set of routes
        
        model.SELECTION = pe.Var(model.R, domain=pe.Binary) #if a route is selected or not
        
        def objective_function(model):
            return sum([self.c_p[names[j]]*model.SELECTION[j] for j in model.R])
        model.OBJECTIVE = pe.Objective(rule=objective_function, sense=pe.minimize)
        
        def demand_satisfaction(model, i):
            return sum([self.a_ij[i,names[j]]*model.SELECTION[j] for j in model.R]) == 1 
        model.con_demand_satisfaction = pe.Constraint(model.V, rule=demand_satisfaction)
        
        solver_results = solver.solve(model, tee=False)
        
        results2 = [{"id": j,
                     "route":names[j],
                    "value": model.SELECTION[j]()}
               for (j) in model.R]
        
        results2 = pd.DataFrame(results2)
        results2 = results2[results2["value"] >= 0.3]
        
        return results2
        
    def _swap_2opt(self, route, i, k):    
        """ Swapping the route """
        new_route = route[0:i]
        new_route.extend(reversed(route[i:k + 1]))
        new_route.extend(route[k + 1:])
        return new_route
        
    def TwoOpt_changes(self):
        
        if self.flag_saving == True:
            
            for i in range(1, len(self.giant) - 1):
                for k in range(i + 1, len(self.giant) - 2):
                                
                    Actual = self.cij[self.giant[i-1], self.giant[i]] + self.cij[self.giant[k], self.giant[k+1]]
                    New = self.cij[self.giant[i-1], self.giant[k]] + self.cij[self.giant[i], self.giant[k+1]]
                                
                    ahorro = New - Actual
                    
                    if ahorro < 0:
                        self.giant = self._swap_2opt(self.giant, i, k)
                        self.add_routes_following_order()
        else:
            
            for i in range(1, len(self.giant) - 1):
                for k in range(i + 1, len(self.giant) - 2):
                    
                    self.giant = self._swap_2opt(self.giant, i, k)
                    self.add_routes_following_order()
                    
                    
    def add_routes_following_order(self):
        
        for i in range(len(self.giant)-2):
            for j in range(i+1, len(self.giant)-1): 
                info_ruta = {}   
                
                if j - i == 1:
                    camino = [0, self.giant[j] ,0]
                    
                    if tuple(camino) not in self.pool:
                        
                        costo = sum(self.cij[camino[k], camino[k+1]] for k in range(len(camino)-1))
                        cap = self.demand[camino[1]]
                        info_ruta[camino[1]] = 1 #la posicion de cada proveedor en la ruta
                
                        if cap <= self.load_capacity and costo <=self.duration:
                            key = tuple(camino) 
                            
                            self.pool[key] = [camino, cap, costo, info_ruta]
                            
                            self.c_p[key] = costo
                            
                            for k in self.demand.keys():
                                self.a_ij[k,key] = 0
                                    
                            self.a_ij[camino[1],key] = 1
                            
                        else:
                            break
                        
                elif j - i >1:
                    
                    if i ==0:
                        camino=self.giant[i:j+1]
                        camino.append(0)
                    else:
                        camino=self.giant[i+1:j+1]
                        camino.append(0)
                        camino.insert(0,0)
                        
                    camino_c = camino.copy()
                    camino_c.reverse()
                    
                    if tuple(camino) not in self.pool and tuple(camino_c) not in self.pool: 
                
                        costo = sum(self.cij[camino[k], camino[k+1]] for k in range(len(camino)-1))
                        cap = sum(self.demand[camino[k]]  for k in range(1, len(camino)-1))
                        
                        if cap <=self.load_capacity and costo <=self.duration:
                            
                            key = tuple(camino) 
                            
                            self.c_p[key] = costo
                            
                            for k in self.demand.keys():
                                self.a_ij[k,key] = 0
                            
                            for k in range(1,len(camino)-1):
                                info_ruta[camino[k]] = k
                                self.a_ij[camino[k],key] = 1
                                
                            self.pool[key] = [camino, cap, costo, info_ruta]
                            
                        else:
                            break

class LKH:
    def __init__(
        self,
        param_nodes, solucionTPP, t, c, Q, daily_time, lkh_time_limit
    ):
        self.nodes = param_nodes #nodes sorted in ascending order 
        self.solucionTPP = solucionTPP
        self.t = t
        self.demand = {i:self.solucionTPP[self.t][1][i] for i in self.nodes}
        self.cij = c #unit time between (i,j)
        self.load_capacity = Q #veh capacity
        self.duration = daily_time  #max working day 
        self.time_limit = lkh_time_limit
        
        self.len_nodes = len(self.nodes)
        self.path = "c:\\LKH"
        self.num_veh = None
        
        ti = time.localtime()
        self.current_time = time.strftime("%H%M%S", ti)
        
    def upper_bound_vehicles(self):
        t_d = sum([np.ceil(self.demand[i]) for i in self.demand])
        veh_c = np.ceil(t_d/self.load_capacity)
        
        suma = 0
        for i in self.nodes:
            if i != 0:
                suma += (np.ceil(self.cij[0, i]) + np.ceil(self.cij[i, 0]))/self.duration
                
        veh_t = np.ceil(suma)
        
        self.num_veh= int(max(veh_c, veh_t))
                
    def generate_input_file(self):
        
        self.upper_bound_vehicles()
        
        with open(self.path + "\\PROBLEM_DIR"+self.current_time+".txt", "w") as h:
            h.writelines(["NAME: INSTANCE\n"])
            
            if self.num_veh >1:
                h.writelines(["TYPE: DCVRP\n"])
            else:
                h.writelines(["TYPE: ATSP\n"])
                
            h.writelines(["DIMENSION: "+str(self.len_nodes)+"\n"])
            h.writelines(["EDGE_WEIGHT_TYPE: EXPLICIT\n"])
            h.writelines(["EDGE_WEIGHT_FORMAT: FULL_MATRIX\n"])
            h.writelines(["CAPACITY: "+str(self.load_capacity)+"\n"])
            h.writelines(["DISTANCE: "+str(self.duration)+"\n"])
            h.writelines(["SERVICE_TIME: "+str(0)+"\n"])
            h.writelines(["VEHICLES : "+str(self.num_veh)+"\n"])
            h.writelines(["EDGE_WEIGHT_SECTION\n"])
            
            for i in range(self.len_nodes):
                h.writelines(["\t"])
                for j in range(self.len_nodes):
                    if i!=j:
                        h.writelines([str(int(self.cij[self.nodes[i],self.nodes[j]]))+"\t"])
                    else:
                        h.writelines([str(0)+"\t"])
                        
                h.writelines(["\n"])
            
            h.writelines(["DEMAND_SECTION\n"])
            
            for i in range(self.len_nodes):
                h.writelines([str(i+1)+"\t"+str(int(self.demand[self.nodes[i]]))+"\n"])
                
            h.writelines(["DEPOT_SECTION\n"])
            h.writelines([str(1)+"\n"])
            h.writelines(["EOF\n"])
            
        with open(self.path + "\\PARAMETER_FILE"+self.current_time+".txt", "w") as h:
            h.writelines(["PROBLEM_FILE = PROBLEM_DIR"+self.current_time+".txt\n"])
            h.writelines(["OUTPUT_TOUR_FILE = PROBLEM_DIR_OUTPUT"+self.current_time+".txt\n"])
            #h.writelines(["TOUR_FILE = TOURPROBLEM_DIR_OUTPUT.txt\n"])
            h.writelines(["TIME_LIMIT = "+str(self.time_limit)+"\n"])
            h.writelines(["EOF\n"])
            
    def execute_LKH(self):
        os.chdir(self.path)
        llamado = self.path + "\\LKH.exe PARAMETER_FILE"+self.current_time+".txt"
        os.system(llamado)
        
    def extract_routes(self):
        
        Rutas_finales = {}
        Rutas_finales[self.t] = {}
        
        if self.len_nodes >2:  
            
            routes = {}
            #crear en el caso de un solo  vehiculo
            f = open(self.path+"\\PROBLEM_DIR_OUTPUT"+self.current_time+".txt", "r")
            
            for i in range(7):    
                garbaje = f.readline()
            
            route = [0]
            indices = {}
            FO_route = 0
            load_route = 0
            time_route = 0
            FO_total = 0
            pos = 1 
            
            node_i = 1
            node_j = next(f).split()[-1]
            
            while node_j != 'EOF':
                
                node_j = int(node_j)
        
                if node_j <= self.len_nodes and node_j != -1:
                    
                    route.append(self.nodes[node_j-1])
                    
                    indices[self.nodes[node_j-1]] = pos
                    
                    self.solucionTPP[self.t][4][self.nodes[node_j-1]] = len(routes)
                    self.solucionTPP[self.t][5][self.nodes[node_j-1]] = pos
                    
                    FO_route += self.cij[self.nodes[node_i-1], self.nodes[node_j-1]]
                    load_route += self.demand[self.nodes[node_j-1]]
                    time_route += self.cij[self.nodes[node_i-1], self.nodes[node_j-1]]
                    
                    node_i = node_j
                    node_j = next(f).split()[-1]
                    pos += 1
                    
                else:
                    
                    index = len(routes)
                    
                    route.append(0)
                    
                    FO_route += self.cij[self.nodes[node_i-1],0]
                    time_route += self.cij[self.nodes[node_i-1],0]
                    
                    routes[index] = [route, load_route, time_route, indices]
                    FO_total += FO_route
                    
                    Rutas_finales[self.t][index] = [route, load_route, time_route, indices]
                    
                    route = [0]
                    FO_route = 0
                    load_route = 0
                    time_route = 0
                    indices = {}
                    pos = 1
                    node_i = 1
                    node_j = next(f).split()[-1]
                    
        else:
            
            FO_total = self.cij[0, self.nodes[1]] + self.cij[self.nodes[1], 0]
            Rutas_finales[self.t][0] = [ [0, self.nodes[1], 0], self.demand[self.nodes[1]], self.cij[0, self.nodes[1]] + self.cij[self.nodes[1], 0], {self.nodes[1]:1} ]
            
            self.solucionTPP[self.t][4][self.nodes[1]] = 0
            self.solucionTPP[self.t][5][self.nodes[1]] = 1
                
                        
        return FO_total, Rutas_finales, self.solucionTPP
        
    def run_LKH(self):
        
        if self.len_nodes >2: 
            
            self.generate_input_file()
            self.execute_LKH()
            
        FO, Routes, solutionTPP_ = self.extract_routes()
        
        return FO, Routes, solutionTPP_

class Graph: # Class to represent a graph
 
    def __init__(self, vertices):
        self.V = vertices # No. of vertices
        self.graph = []
 
    # function to add an edge to graph
    def addEdge(self, u, v, w):
        self.graph.append([u, v, w])
         
    # utility function used to print the solution
    def printArr(self, dist, etiqueta):
        print("Vertex Distance from Source")
        for i in range(self.V):
            print("{0}\t\t{1}\t\t{2}".format(i, dist[i], etiqueta[i]))
     
    # The main function that finds shortest distances from src to
    # all other vertices using Bellman-Ford algorithm. The function
    # also detects negative weight cycle
    def BellmanFord(self, src):
 
        # Step 1: Initialize distances from src to all other vertices
        # as INFINITE
        dist = [float("Inf")] * self.V
        etiqueta = [0]*self.V
        dist[src] = 0
        
        # Step 2: Relax all edges |V| - 1 times. A simple shortest
        # path from src to any other vertex can have at-most |V| - 1
        # edges
        for _ in range(self.V - 1):
            # Update dist value and parent index of the adjacent vertices of
            # the picked vertex. Consider only those vertices which are still in
            # queue
            for u, v, w in self.graph:
                if dist[u] != float("Inf") and dist[u] + w < dist[v]:
                        dist[v] = dist[u] + w
                        etiqueta[v] = u
        
        # print all distance
        #self.printArr(dist, etiqueta)
        
        Cabeza = -1
        Cola = self.V-1
        FO = dist[Cola]
        arcos = []
        while Cabeza !=0 :
            Cabeza = etiqueta[Cola]
            arcos.append((Cabeza,Cola))
            Cola = Cabeza
        
        #print(FO)
        #print(arcos)
        return FO, arcos
    
def ejecuta_vecino_mas_cercado(nodes, t, max_cij, c):
    
    Info_Route = {}
    
    if len(nodes)>1: 
        
        Info_Route[t] = [nodes, len(nodes)]
        
        matrix_cij = np.full((Info_Route[t][1],Info_Route[t][1]), max_cij) 
        
        #Build distance matrix just for supplier selected
        for i in range(len(Info_Route[t][0])):
            for j in range(len(Info_Route[t][0])):
                if i != j :
                    matrix_cij[i,j] = c[Info_Route[t][0][i],Info_Route[t][0][j]]
                    
        #Run nearest neighbor algorithm
        Ruta = []            
        Cabeza = 0 
        Ruta.append(Cabeza)
        
        while len(Ruta) < Info_Route[t][1]:
            minimo = max_cij
            for i in range(len(nodes)):
                if matrix_cij[Cabeza][i] < minimo:
                    minimo = matrix_cij[Cabeza][i]
                    Cola = i
                
                matrix_cij[i][Cabeza]=max_cij
                
            Ruta.append(Info_Route[t][0][Cola])
            Cabeza = Cola
            
        Info_Route[t].append(Ruta) #General route
    
    return Info_Route
    
#Build Augmented graph following general tour
def ejecuta_splitProcedure(Info_Route, solucionTTP, t, c, Q):
    
    Rutas_finales = {}
    FO_Rutas = {}
    Rutas_finales[t] = {}
    info = {}
    
    if len(Info_Route) > 0: 
        g = Graph(Info_Route[t][1]) #Define the number of suppliers
        
        #Build the arcs just if they respect vehicle capacity following the general route found
        for i in range(Info_Route[t][1]-1):
            for j in range(i+1, Info_Route[t][1]): 
                info_ruta = {}   
                
                if j - i == 1:
                    camino = [0, Info_Route[t][2][j] ,0]
                    costo = sum(c[camino[k], camino[k+1]] for k in range(len(camino)-1))
                    cap = solucionTTP[t][1][camino[1]]
                    info_ruta[camino[1]] = 1
            
                    if cap <= Q and costo <=daily_time:
                        info[(i,j)] = [camino, cap, costo, info_ruta]
                        g.addEdge(i,j,costo)
                        
                    else:
                        break
                        
                elif j - i >1:
                    if i ==0:
                        camino=Info_Route[t][2][i:j+1]
                        camino.append(0)
                    else:
                        camino=Info_Route[t][2][i+1:j+1]
                        camino.append(0)
                        camino.insert(0,0)
                
                    costo = sum(c[camino[k], camino[k+1]] for k in range(len(camino)-1))
                    cap = sum(solucionTTP[t][1][camino[k]]  for k in range(1, len(camino)))
                    
                    if cap <=Q and costo <=daily_time:
                        
                        for k in range(1,len(camino)-1):
                            info_ruta[camino[k]] = k
                            
                        info[(i,j)] = [camino, cap, costo, info_ruta]
                        g.addEdge(i,j,costo)
                        
                    else:
                        break
        
        FO_Routing, Arcos_agregados = g.BellmanFord(0) #Run BellamnFord Algorithm 
        
        #Translate arc in routing solution
        for i in range(len(Arcos_agregados)):
            Rutas_finales[t][i] = info[Arcos_agregados[i]]
            for k in info[Arcos_agregados[i]][3]:
                solucionTTP[t][4][k]=i
                solucionTTP[t][5][k] = info[Arcos_agregados[i]][3][k]
                
    else:
        FO_Routing = 0
            
    return FO_Routing, Rutas_finales


def upper_bound_vehicles(nodes, demand, Q, duration, cij):
    
    t_d = sum([np.ceil(demand[i]) for i in demand])
    veh_c = np.ceil(t_d/Q)

    veh_t = np.ceil(sum((np.ceil(cij[0, i]) + np.ceil(cij[i, 0]))/duration  for i in nodes if i != 0))
    
    veh = int(max(veh_c, veh_t)*2)
    
    return veh

def solve_DC_CVRP(nodes, d, num_veh, cij, cap, l_max):
    
    F = range(num_veh)
    V = nodes
    
    M = nodes.copy()
    M.remove(0)
    
    c_ = {(i,j,v):0 for (i,j) in cij.keys() for v in F}
    
    m = gu.Model('dc-cvrp')
    
    x = m.addVars(c_.keys(), vtype=gu.GRB.BINARY, name="x_")
    w = {(i,v):m.addVar(vtype=gu.GRB.BINARY, name="w_"+str((i,v))) for v in F for i in M}
    u = {(i,v):m.addVar(vtype=gu.GRB.CONTINUOUS, name="u_"+str((i,v))) for v in F for i in M}
    
    #Each customer is visited exactly once
    m.addConstrs((gu.quicksum(w[i, v] for v in F) == 1 for i in M), name="visit_once")
    
    #Capacity constraint for each vehicle
    m.addConstrs((gu.quicksum(d[i] * w[i, v] for i in M) <= cap for v in F), name="capacity_constraint")
    
    #Non-split
    m.addConstrs((gu.quicksum(w[i,v] for v in F ) <= 1 for i in M), name="Non-split")
    
    #If a vehicle travels from i to j, w_i^v = 1 and w_j^v = 1
    m.addConstrs((x[i, j, v] <= w[i, v] for i in M for j in M for v in F if i!=j), name="customer_assignment_1")
    m.addConstrs((x[i, j, v] <= w[j, v] for i in M for j in M for v in F if i!=j), name="customer_assignment_2")
    
    #If a vehicle visits a customer, it leaves that customer
    m.addConstrs((x.sum(hh,'*',v) ==  w[hh,v] for v in F for hh in M), name="leaves_part_1")
    m.addConstrs((x.sum('*',hh,v) ==  w[hh,v] for v in F for hh in M), name="leaves_part_2")
    
    #max time limit per route
    m.addConstrs((gu.quicksum(cij[i,j]*x[i,j,v] for (i,j) in cij.keys()) <= l_max for v in F), name="max_time_per_route")
             
    #start inequality
    m.addConstrs((w[i,v] <= gu.quicksum(x[0,j,v] for j in M) for v in F for i in M), name="start_inequality")
    
    #avoid sub tours
    m.addConstrs((u[i,v] - u[j,v] + len(V)*x[i,j,v] <= len(V)-1  for v in F for i in M for j in M if i!= j), name="subtour_avoidance")
    
    #symmetry
    m.addConstrs((gu.quicksum(x[0, j,v]  for j in M) <= gu.quicksum(x[0, j, v-1]  for j in M)  for v in F if v >0), name="symmetry")
    
    #Objective function
    m.setObjective(gu.quicksum(cij[i,j]*x[i,j,v] for (i,j) in cij.keys() for v in F))
    
    m.setParam('OutputFlag',0)
    m.update()
    m.optimize()
    
    if m.Status==2 or m.Status==9:
        
        x_sol = [(i,j,v) for i,j,v in x if x[i,j,v].x >.5]
        Rutas = {}
        FO_global = 0
        
        for v in F:
                
            x_sol_v = [(i[0],i[1]) for i in x_sol if i[2]==v]
            
            if len(x_sol_v) >0 :
            
                time = 0
                cap = 0
                tour = [0]
                
                index = 1
                info_ruta = {}
                
                for _ in range(len(x_sol_v)):
                    
                    arc = [j for (i,j) in x_sol_v if i==tour[-1]]
                    time += cij[tour[-1],arc[0]]
                    
                    if arc[0] != 0:
                        
                        cap += d[arc[0]]
                        info_ruta[arc[0]] = index
                        index += 1
                        
                    tour.append(arc[0])
                    
                Rutas[v] = [tour, round(cap,2), round(time,2), info_ruta]
                
                FO_global += time
                
            
        return FO_global, Rutas
    
    else:
        breakpoint()
    

def exact_model_VRP(nodes, solucionTTP, t, c, Q, daily_time):
    
    Rutas_finales = {}
    
    demand = {i:solucionTTP[t][1][i] for i in nodes}
    num_veh = upper_bound_vehicles(nodes, demand, Q, daily_time, c)
    
    matrix_cij = {(i,j):c[i,j]  for i in nodes for j in nodes if i != j }
    
    FO, routes = solve_DC_CVRP(nodes, demand, num_veh, matrix_cij, Q,  daily_time)
    
    Rutas_finales[t] = routes
    
    for i in range(len(Rutas_finales[t])):
        for k in Rutas_finales[t][i][3]:
            solucionTTP[t][4][k]=i
            solucionTTP[t][5][k] = Rutas_finales[t][i][3][k]
    
    return FO, Rutas_finales, solucionTTP
    
def Genera_ruta_at_t(solucionTTP, t, max_cij, c, Q, flag_):
    
    nodes = [i for i, x in enumerate(solucionTTP[t][0]) if x] #See which suppliers are in the solution
    nodes.insert(0,0)
    
    Rutas_finales = {}
    Rutas_finales[t] = {}

    if len(nodes) > 1:
        
        if flag_  == 'SPLIT':
            
            Info_Route = ejecuta_vecino_mas_cercado(nodes, t, max_cij, c)
            FO_Routing, Rutas_finales = ejecuta_splitProcedure(Info_Route, solucionTTP, t, c, Q) 
            
        elif flag_ == 'LKH':
            
            alg = LKH(nodes, solucionTTP, t, c, Q, daily_time, lkh_time_limit)
            FO_Routing, Rutas_finales, solucionTTP = alg.run_LKH()
            
        elif flag_ == 'SP':
            
            Info_Route = ejecuta_vecino_mas_cercado(nodes, t, max_cij, c)
            
            nodes = Info_Route[t][2]
            nodes.append(0)
            
            alg = Set_Partitioning(nodes, solucionTTP, t, c, Q, daily_time, solver_name)
            FO_Routing, Rutas_finales, solucionTTP = alg.Run_SP_TwoOPT()
            
        elif flag_ == 'EXACT':
            
            FO_Routing, Rutas_finales, solucionTTP = exact_model_VRP(nodes, solucionTTP, t, c, Q, daily_time)
            
    else:
        
        FO_Routing = 0
        
    FO_Routing = FO_Routing*increase_route
    
    return Rutas_finales, solucionTTP, FO_Routing

def Grafica_Ruta(coor, N, Route=[]):
    if len(Route) > 0:
        G = nx.DiGraph(Route)
    else:
        G = nx.DiGraph()
    G.add_nodes_from(N)
    options = {
        "font_size": 13,
        "node_size": 250,
        "node_color": "white",
        "edgecolors": "black",
        "linewidths": 0.5,
        "width": 0.5,
    }
    
    fig, ax = plt.subplots(figsize=(10, 8))
    nx.draw_networkx(G, coor, **options)
    # Set margins for the axes so that nodes aren't clipped
    ax = plt.gca()
    ax.margins(0.20)
    plt.axis('on')
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    #plt.xlabel('Longitud') 
    #plt.ylabel('Latitud')
    plt.title('Gammas according to distance estimation') 
    plt.show()

def Crea_Grafica_Rutas_dia_t(final_policy, t, coor, N):
    Route = []
    for ruta in final_policy[t][0][9][t].keys():
        #print("Ruta"+str(ruta))
        for i in range( len(final_policy[t][0][9][t][ruta][0])-1):
            #print((final_policy[t][0][9][0][ruta][0][i],final_policy[t][0][9][0][ruta][0][i+1]))
            Route.append((final_policy[t][0][9][t][ruta][0][i],final_policy[t][0][9][t][ruta][0][i+1]))  
    
    Grafica_Ruta(coor, N, Route)
    
def Build_indicators_policy(nombre, lower, upper, info_gammas, H, policy, alpha, V, K, per_por, T, M, coor, Mk, Q, d, d_EV, q, q_EV, p, p_EV, c, max_cij, dist_demand_parm, r_EV, bo_EV, Km, dist_p, dist_q, best_theta, flag, d_log, p_log, q_log, seed = None):
    print(nombre)
    
    if seed is not None:    
        np.random.seed(seed)
    
    
    if len(info_gammas) == 1:
        gamma_v = [info_gammas[0] for i in M]
    else:
        gamma_v = info_gammas.copy() 
    
    objetive_funtion = np.zeros((upper-lower, len(T)))
    quantity_purchased = np.zeros((upper-lower, len(T)))
    inventory_level = np.zeros((upper-lower, len(T)))
    routing_cost = np.zeros((upper-lower, len(T)))
    profit = np.zeros((upper-lower, len(T)))
    revenue = np.zeros((upper-lower, len(T)))
    purchase_cost = np.zeros((upper-lower, len(T)))
    backorders_cost = np.zeros((upper-lower, len(T)))
    no_suppliers_per_time = np.zeros((upper-lower, len(T))) 
    no_routes_per_time = np.zeros((upper-lower, len(T)))
    backorders_unit = np.zeros((upper-lower, len(T)))
    
    if type(best_theta) == list:
        best_theta1 = best_theta[0]
        best_theta2 = best_theta[1]
    
    R_aux = []
    time_ = []
    gap_ = []
    other_indicators = {}
    explicit_solutions = {}
        
    for sample in range(lower, upper):
        
        tiempoInicio = time.time()
        
        if policy== 0: #Corre mipic
            final_policy, indicators = Deterministic_Rolling_Horizon(Vertex, V, Products, K, per_por, Periods, T, M, Mk, Q, d, d_EV,q, q_EV,p, p_EV, c, max_cij, sample, H, gamma_v, r_EV, bo_EV, alpha, Km, best_theta, MIP_gap, flag)
            R_aux.append(indicators[0])
            
        elif policy== 1: #Corre deterministic
            final_policy, indicators = Deterministic_Rolling_Horizon(Vertex, V, Products, K, per_por, Periods, T, M, Mk, Q, d, d_EV,q, q_EV,p, p_EV, c, max_cij, sample, H, gamma_v, r_EV, bo_EV, alpha, Km, 0, MIP_gap, flag)
            R_aux.append(indicators[0])
            
        elif policy == 2:#Corre stochastic
            final_policy, indicators, count_i, gamma_i = Stochasctic_Rolling_Horizon(Vertex, V, Products, K, per_por, Periods, T, M, Mk, Q, d, dist_demand_parm, q, dist_q, p, dist_p, c, max_cij, sample, sample_paths, H, gamma_v, r_EV, bo_EV, alpha, MIP_gap, d_log, p_log, q_log)
            R_aux.append(indicators[0])
            
        elif policy == 3: #Corre stochastic Xij explicit
            final_policy, indicators, count_i, gamma_i = Stochasctic_Rolling_Horizon_xij(Vertex, V, Products, K, per_por, Periods, T, M, Mk, Q, d, dist_demand_parm, q, dist_q, p, dist_p, c, max_cij, sample, sample_paths, H, gamma_v, r_EV, bo_EV, alpha, MIP_gap, d_log, p_log, q_log, Km)
            R_aux.append(indicators[0])
            
        elif policy == 4: #Corre stochastic with Daganzo
            final_policy, indicators, count_i, gamma_i = Stochasctic_Rolling_Horizon_Daganzo(Vertex, V, Products, K, per_por, Periods, T, M, Mk, Q, d, dist_demand_parm, q, dist_q, p, dist_p, c, max_cij, sample, sample_paths, H, gamma_v, r_EV, bo_EV, alpha, MIP_gap, d_log, p_log, q_log, coor, best_theta1, best_theta2)
            R_aux.append(indicators[0])
        
        tiempoOpti = time.time()-tiempoInicio
        time_.append(tiempoOpti)
        #gap_.append(indicators[8])
        
        other_indicators[sample] = indicators.copy()
        explicit_solutions[sample] = final_policy.copy()
        
        for t in T:
            
            quantity_purchased[sample-lower, t] =sum(final_policy[t][0][6])
            inventory_level[sample-lower, t] = sum(final_policy[t][1][1])
            routing_cost[sample-lower, t] = final_policy[t][3]
            profit[sample-lower, t] = final_policy[t][4]
            revenue[sample-lower, t] = final_policy[t][5]
            purchase_cost[sample-lower, t] = final_policy[t][6]
            backorders_cost[sample-lower, t] = final_policy[t][7]
            no_suppliers_per_time[sample-lower, t] = sum(final_policy[t][0][0])
            no_routes_per_time[sample-lower, t] = len(final_policy[t][0][9][t])
            objetive_funtion[sample-lower, t] = final_policy[t][8]
            backorders_unit[sample-lower, t] = sum(final_policy[t][2])
            
            if T_aux == True:
                break
        
    indicators = [quantity_purchased, inventory_level, routing_cost, profit, revenue, purchase_cost, backorders_cost, no_suppliers_per_time, no_routes_per_time, objetive_funtion, backorders_unit, other_indicators, explicit_solutions]
    
    EV_OBJ = round(np.average([sum(objetive_funtion[sample]) for sample in range(upper-lower)]),2)
    time_ = round(np.average(time_),2)
    #gap_ =  round(np.mean(gap_),2)
    gap_ =  'nada'
    #print(EV_OBJ, np.average(R_aux))
    print(EV_OBJ, time_)
    
    nombre2 = nombre+"_"+str(lower)+","+str(upper)
    
    #with open(nombre2+'.pickle', 'wb') as handle:
    #    pickle.dump(indicators, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return (EV_OBJ, time_,gap_), indicators


def genera_espejo_valor(level, valor_referencia, valor):
    
    if level == 0: #por abajo
    
        if valor > valor_referencia:
            
            valor = valor_referencia + (valor_referencia - valor)
        
        
    elif level == 1: #por arriba
    
        if valor < valor_referencia:
            
            valor = valor_referencia + (valor_referencia - valor)
            
    return valor
    
          
def provides_random_value(left_truncated_value, dist_param, distribution, l = None, EV_valor = None, tipo=None):
    
    #bandera = False
    #while bandera == False:
        
    if distribution == 'Norm':
        
        valor = np.random.normal(dist_param[0],dist_param[1])
        
        #if left_truncated_value <= valor and valor <= 2*mean-left_truncated_value:
            #bandera = True
        #else:
        #    print('se salio de los limites')
        
    elif distribution == 'LogNorm':
        
        valor = np.random.lognormal(dist_param[0],dist_param[1])
            
    elif distribution == 'Triang':
        
        valor = np.random.triangular(dist_param[0],dist_param[2], dist_param[1])
        
    elif distribution == 'Uniform':
        
        valor = np.random.uniform(dist_param[0],dist_param[1])
        
    
    if prices_correlated_time == True and offer_correlated_time == False and tipo == 'price':
        
        valor = genera_espejo_valor(l, EV_valor, valor)
              
        
    elif prices_correlated_time == False and offer_correlated_time == True and tipo == 'offer':
        
        valor = genera_espejo_valor(l, EV_valor, valor)
        
        
    elif prices_correlated_time == True and offer_correlated_time == True and (tipo == 'price' or tipo == 'offer'):
        
        valor = genera_espejo_valor(l, EV_valor, valor)
        
    return valor   

def Genera_Instacias_Stochastic_with_parameters_fixed(Vertex, Products, Periods, paths_evaluation, veh_factor, cap_factor, seedd, increase_in_route, increase_offering, perish_rate, name_exp):
    
    V = range(Vertex) #Set of vertexs M U {0} Depot
    M = range(1, Vertex) #Set of farmers
    K = range(Products)  #Set of products
    T = range(Periods) #Set of periods
    per_por = {k:perish_rate for k in K}
    Q = veh_factor
    
    if crea_datos == True:
        
        np.random.seed(seedd)
    
        #Coordenadas (x,y) de las ciudades
        coor = {i:(np.random.randint(0, size_grid), np.random.randint(0, size_grid)) for i in M}
        coor[0] = (size_grid/2, size_grid/2)
        
        Mk = {}
        Km = {i:[] for i in M}    
        
        t = 0
        Markets = []
        for k in K:
            if (k,t) not in Mk.keys():
                Mk[k,t]=[]
            #MarkesLen = np.random.randint(np.ceil(Vertex*0.5), Vertex)
            
            MarkesLen = np.ceil((Vertex-1)*increase_offering)
            M_ayuda = [i for i in M]
            np.random.shuffle(M_ayuda)
            
            while len(Mk[k,t]) < MarkesLen:
                #seleccion = np.random.randint(1, Vertex)
                seleccion = M_ayuda[len(Mk[k,t])]
                
                if seleccion not in Mk[k,t]:
                    Mk[k,t].append(seleccion)
                    Km[seleccion].append(k)
                    if seleccion not in Markets:
                        Markets.append(seleccion)
                        
        for i in M:
            if i not in Markets:
                selecion = np.random.randint(Products)
                Mk[selecion,t].append(i)
                Km[i].append(selecion)
        
        for k in K:
            Mk[k,t].sort()
            
        for t in range(1, len(T)):
            for k in K:
                Mk[k,t] = Mk[k,0].copy()
                
        p = {}
        q = {}           
    
        p_EV = {}
        q_EV = {}
        
        r_EV = {}
        bo_EV = {}
        
        EV_price = (upper_price + lower_price)/2.0
        
        if flag_correlated_price == True:
            dist_p = {}
            avg_EV_p = {}
            
            
            for k in K:
                ayuda = []
                for i in Mk[k,0]:   
                    if k == Km[i][0]:
                        bandera = False 
                        first =  np.random.uniform(lower_price, upper_price)
                        if first <= EV_price:
                            bandera = True
                            
                        dist_p[i,k] = (first, first*std_price, bandera)
                        
                    else:
                        value = np.random.uniform(lower_price, upper_price)
                        
                        if dist_p[i,Km[i][0]][2] == True:
                            if value <= EV_price:
                                dist_p[i,k] = (value, value*std_price)
                            else:
                                numero = EV_price + (EV_price - value)
                                dist_p[i,k] = (numero, numero*std_price)
                        else:
                            if value >= EV_price:
                                dist_p[i,k] = (value, value*std_price) 
                            else:
                                numero = EV_price + (EV_price - value)
                                dist_p[i,k] = (numero, numero*std_price)
                                
                    ayuda.append(dist_p[i,k][0])
                    
                avg_EV_p[k] = np.average(ayuda)
        else:
            dist_p = {}
            avg_EV_p = {}
            for k in K:
                ayuda = []
                for i in Mk[k,0]:
                    valor = np.random.uniform(lower_price, upper_price)
                    dist_p[i,k] = (valor, valor*std_price)
                    ayuda.append(valor)
                avg_EV_p[k] = np.average(ayuda)
                
        new_dist_p = generate_new_distribution_parameters(dist_p, dist_train)
        
        EV_cap = (upper_quiantity + lower_quiantity)/2.0
        
        if flag_correlated_cap == True:
            dist_q = {}
            avg_EV_q = {}
            
            for k in K:
                ayuda = []
                for i in Mk[k,0]:   
                    if k == Km[i][0]:
                        bandera = False 
                        first =  np.random.uniform(lower_quiantity, upper_quiantity)
                        if first <= EV_cap:
                            bandera = True
                            
                        dist_q[i,k] = (first, first*std_quiantity, bandera)
                        
                    else:
                        value = np.random.uniform(lower_quiantity, upper_quiantity)
                        
                        if dist_q[i,Km[i][0]][2] == True:
                            if value <= EV_cap:
                                dist_q[i,k] = (value, value*std_quiantity)
                            else:
                                numero = EV_cap + (EV_cap - value)
                                dist_q[i,k] = (numero, numero*std_quiantity)
                        else:
                            if value >= EV_cap:
                                dist_q[i,k] = (value, value*std_quiantity) 
                            else:
                                numero = EV_cap + (EV_cap - value)
                                dist_q[i,k] = (numero, numero*std_quiantity)
                                
                    ayuda.append(dist_q[i,k][0])
                avg_EV_q[k] = np.average(ayuda)
    
        else:
            dist_q = {}
            avg_EV_q = {}
            
            for k in K:
                ayuda = []
                for i in Mk[k,0]:
                    valor = np.random.uniform(lower_quiantity, upper_quiantity)*cap_factor
                    dist_q[i,k] = (valor, valor*std_quiantity)
                    ayuda.append(valor)
                    
                avg_EV_q[k] = np.average(ayuda)
                
        new_dist_q = generate_new_distribution_parameters(dist_q, dist_train)

        for k in K:
            for t in T:
                for path_eval in range(paths_evaluation):
                    
                    if prices_correlated_time == True or offer_correlated_time == True:
                        
                        l_selected = np.random.randint(0,levels)
                        
                    else:
                        l_selected = -1
                    
                    for i in Mk[k,0]:
                        #Expected value of the price and availability according to the stipulated limits.
                        p_EV[i,k] = dist_p[i,k][0]
                        q_EV[i,k] = dist_q[i,k][0]
                
                
                        if prices_correlated_time == True and offer_correlated_time == True:
                            
                            if l_selected == 0:
                                
                                l_price = 1
                                l_offer = 0
                                
                            elif l_selected == 1:
                                
                                l_price = 0
                                l_offer = 1
                                
                            elif l_selected == 2:
                                
                                l_price = 2
                                l_offer = 2
                
                    #Real values of price and availability for each path evaluation (This is for policy evaluation)
                            p[i,k,t,path_eval]= provides_random_value(1, new_dist_p[i,k], dist_train, l_price, EV_price, 'price')
                            q[i,k,t,path_eval] = provides_random_value(0, new_dist_q[i,k], dist_train, l_offer, EV_cap, 'offer')
                            
                        else:
                            
                            p[i,k,t,path_eval]= provides_random_value(1, new_dist_p[i,k], dist_train, l_selected, EV_price, 'price')
                            q[i,k,t,path_eval] = provides_random_value(0, new_dist_q[i,k], dist_train, l_selected, EV_cap, 'offer')
                        
                            
                            
            r_EV[k] = np.average([p_EV[i,k] for i in Mk[k,t]])*(1+revenue_dist)
            bo_EV[k] = np.average([p_EV[i,k] for i in Mk[k,t]])*2.5*100
        
        
        #Demand distribution is built
        dist_demand_parm = {}
        
        for k in K:
            mean = np.random.uniform(lower_demand, upper_demand)
            dist_demand_parm[k] = (mean, mean*std_demand)
            
        new_dist_d = generate_new_distribution_parameters(dist_demand_parm, dist_train)
        
        d = {}
        d_EV = {}
    
        for k in K:
            for t in T:
                d_EV[k,t] = dist_demand_parm[k][0]
                
                for path_eval in range(paths_evaluation):
                    
                    if flag_demand_cero_t_0 == True:
                        if t == 0:
                            
                            d[k,t,path_eval]  = 0
                        else: 

                            d[k,t,path_eval] = provides_random_value(0, new_dist_d[k], dist_train)
                                    
                    else:
                        
                        d[k,t,path_eval] = provides_random_value(0, new_dist_d[k], dist_train)
        
        #Travel cost between (i,j). It is the same for all time t
        c = {}
        for i in range(Vertex):
            for j in range(Vertex):
                if i !=j:
                    dist = round(np.sqrt(((coor[i][0]-coor[j][0])**2)+((coor[i][1]-coor[j][1])**2)),3)
                    if j != 0:
                        dist += time_per_supplier
                    
                    c[i,j] = dist
                    
        max_cij = max(c.values())*2
        
        datos = {}
        datos['coor'] = coor.copy()
        datos['Mk'] = Mk.copy()
        datos['Km'] = Km.copy()
        
        datos['d'] = d.copy()
        datos['d_EV'] = d_EV.copy()
        datos['q'] = q.copy()
        datos['q_EV'] = q_EV.copy()
        datos['p'] = p.copy()
        datos['p_EV'] = p_EV.copy()
        datos['c_ij'] = c.copy()
        datos['max_cij'] = max_cij
        datos['dist_d'] = dist_demand_parm.copy()
        datos['r_EV'] = r_EV.copy()
        datos['bo_EV']  = bo_EV.copy()
        datos['dist_p'] = dist_p.copy()
        datos['dist_q'] = dist_q.copy()
        datos['avg_EV_p'] = avg_EV_p.copy()
        datos['avg_EV_q'] = avg_EV_q.copy()
        
        datos['new_dist_p'] = new_dist_p.copy() 
        datos['new_dist_q'] = new_dist_q.copy()
        datos['new_dist_d'] = new_dist_d.copy()
        
        with open('Datos'+name_exp+'.pickle', 'wb') as handle:
            pickle.dump(datos, handle, protocol=pickle.HIGHEST_PROTOCOL)
             
    else:
        
        if dist_train == dist_eval:
            
            with open('Datos'+name_exp+'.pickle', 'rb') as handle:
                datos = pickle.load(handle)
                
            d = datos['d']
            q = datos['q'] 
            p = datos['p'] 
            
        else:
            
            split_1 = re.split('_', name_exp)
            split_2 = re.split('x', split_1[1])
            
            #cargo los datos de train
            
            with open('Datos'+split_1[0]+'_'+split_2[0]+'x'+split_2[0]+'.pickle', 'rb') as handle:
                datos = pickle.load(handle)
            
            #cargo los datos de test
            with open('Datos'+split_1[0]+'_'+split_2[1]+'x'+split_2[1]+'.pickle', 'rb') as handle:
                datos_test = pickle.load(handle)
                
            d = {}
            q = {}
            p = {}
            
            for key in datos['q']:
                if lo_eval <= key[3] and key[3] <up_eval: #datos de test
                    p[key] = datos_test['p'][key]
                    q[key] = datos_test['q'][key]
                    
                else: #datos de train
                    p[key] = datos['p'][key]
                    q[key] = datos['q'][key]
                    
            for key in datos['d']:
                if lo_eval <= key[2] and key[2] <up_eval: #datos de test
                    d[key] = datos_test['d'][key]
                    
                else: #datos de train
                    d[key] = datos['d'][key]
        
        coor = datos['coor']
        Mk = datos['Mk']
        Km = datos['Km']
        d_EV = datos['d_EV']
        q_EV = datos['q_EV']
        
        p_EV = datos['p_EV'] 
        c = datos['c_ij'] 
        max_cij = datos['max_cij'] 
        dist_demand_parm = datos['dist_d'] 
        r_EV = datos['r_EV'] 
        bo_EV = datos['bo_EV']  
        dist_p = datos['dist_p'] 
        dist_q = datos['dist_q'] 
        avg_EV_p = datos['avg_EV_p'] 
        avg_EV_q = datos['avg_EV_q'] 
        
        new_dist_p = datos['new_dist_p'] 
        new_dist_q = datos['new_dist_q'] 
        new_dist_d = datos['new_dist_d'] 
                
    return V, K, per_por, T, M, coor, Mk, Q, d, d_EV, q, q_EV, p, p_EV, c, max_cij, dist_demand_parm, r_EV, bo_EV, Km, dist_p, dist_q, avg_EV_p, avg_EV_q, new_dist_d, new_dist_p, new_dist_q

def Deterministic_Rolling_Horizon(Vertex, V, Products, K, per_por, Periods, T, M, Mk, Q, d, d_EV,q, q_EV,p, p_EV, c, max_cij, sample, H, gamma, r_EV, bo_EV, alpha, Km, theta, gap, flag_policy):
    
    tiempoInicio = time.time()
    gammas_RH = {i:round(gamma[i-1],3) for i in M}
    
    final_policy = {}
    FO_policy_routing = 0
    FO_policy_profit= 0
    FO_policy_purchase = 0
    FO_policy_bo = 0
    FO_policy_revenue = 0
    FO_policy_obj = 0
    
    
    solucionTTP = {t:[np.zeros(len(V), dtype=bool), np.zeros(len(V), dtype=float), np.zeros((len(V), len(K)), dtype=bool), np.zeros((len(V), len(K)), dtype=float), np.full(len(V) , -1, dtype = int), np.zeros(len(V), dtype=int), np.zeros(len(K), dtype=float), 0, 0] for t in T}
    compra_extra = {t:np.zeros(len(K), dtype = float) for t in T}
    inventario = {t:[[0 for k in K], [0 for k in K]]  for t in range(len(T))}
    
    I_0 = {k:0.0 for k in K} #Initial inventory level
    
    for t_RH in T:
        
        if t_RH + H < len(T)-1:
            
            t_max = t_RH+H
        else:
            t_max=len(T)-1
        
        TT = range(t_RH,t_max+1)
        
        C_MIP = {}
        for i in M:
            for t in TT:
                if t == t_RH:
                    C_MIP[i,t] = (c[0,i]+c[i,0])*gammas_RH[i]
                else:
                    C_MIP[i,t] = (c[0,i]+c[i,0])*gammas_RH[i]
                    
        
        d_model = {}
        q_model = {}
        p_model = {}
        
        for k in K:
            for t in TT:
                if t == t_RH:
                    
                    d_model[k,t] = d[k,t,sample]
                        
                    for i in Mk[k,t]:
                        q_model[i,k,t] = q[i,k,t, sample]
                        p_model[i,k,t] = p[i,k,t, sample]
                        
                else:
                    
                    if flag_policy == False:
                        d_model[k,t] = d_EV[k,t]
                        
                        for i in Mk[k,t]:
                            q_model[i,k,t] = q_EV[i,k]
                            p_model[i,k,t] = p_EV[i,k]
                    else:
                        d_model[k,t] = d[k,t,sample]*theta
                        
                        for i in Mk[k,t]:
                            q_model[i,k,t] = q[i,k,t, sample]
                            p_model[i,k,t] = p_EV[i,k]
                        
        m = gu.Model('Inventory')
        
        #Inventory variables
        z = {(i,k,t):m.addVar(vtype=gu.GRB.CONTINUOUS, name="z_"+str((i,k,t))) for t in TT for k in K for i in Mk[k,t]}
        w = {(i,t):m.addVar(vtype=gu.GRB.BINARY, name="w_"+str((i,t))) for t in TT for i in M}
        ii = {(k,t):m.addVar(vtype=gu.GRB.CONTINUOUS, name="i_"+str((k,t))) for k in K for t in TT}
        bo = {(k,t):m.addVar(vtype=gu.GRB.CONTINUOUS, name="bo_"+str((k,t))) for t in TT for k in K}
        
        #Inventory constrains
        for k in K:
            m.addConstr(ii[k,t_RH] == I_0[k] + gu.quicksum(z[i,k,t_RH] for i in Mk[k,t_RH]) - (d_model[k,t_RH]*(1+0)) + bo[k,t_RH])
            
            #if I_0[k] >= d_model[k,t_RH]:
             #   m.addConstr(ii[k,t_RH] == I_0[k] + gu.quicksum(z[i,k,t_RH] for i in Mk[k,t_RH]) - (d_model[k,t_RH]*(1+0)) + bo[k,t_RH])
            #else:
             #   m.addConstr(ii[k,t_RH] == I_0[k] + gu.quicksum(z[i,k,t_RH] for i in Mk[k,t_RH]) - (d_model[k,t_RH]*(1+theta)) + bo[k,t_RH])
            
        #if H != 0:
         #   for k in K: 
          #      m.addConstr(ii[k,t_max] >= d_model[k,t_max]*0.5)
          
        for t in TT:
            for i in M:
                m.addConstr((c[0,i]+c[i,0])*w[i,t] <= daily_time)
        
        for k in K:            
            for t in TT:
                if t > t_RH:
                    m.addConstr(ii[k,t] == (ii[k,t-1]*(1-per_por[k])) + gu.quicksum(z[i,k,t] for i in Mk[k,t]) - d_model[k,t] + bo[k,t])

        for t in TT:
            for k in K:
                for i in Mk[k,t]:
                    m.addConstr(z[i,k,t] <= q_model[i,k,t]*w[i,t])
                    
        for t in TT:
            for i in M:
                m.addConstr(gu.quicksum(z[i,k,t] for k in K if (i,k,t) in z) <= Q )
                
        purchase = gu.quicksum(p_model[i,k,t]*z[i,k,t] for t in TT for k in K for i in Mk[k,t])
        backorders = gu.quicksum(bo_EV[k]*bo[k,t] for k in K for t in TT)
        route = gu.quicksum(C_MIP[i,t]*w[i,t] for t in TT for i in M)*increase_route
        revenue = gu.quicksum(r_EV[k]*d_model[k,t] for t in TT for k in K)
        
        m.setObjective(((1-alpha)*((-1)*route)) + (alpha)*(revenue-purchase-backorders), sense = gu.GRB.MAXIMIZE)
        m.update()
    
        #m.write('MV-TPP.lp')
        
        m.setParam('OutputFlag',0)
        m.setParam('MIPGap',gap)
        m.optimize()
        
        if m.Status==2 or m.Status==9:
            
            costo_revenue_dia_t = sum(r_EV[k]*d_model[k,t_RH] for k in K)
            costo_compra_dia_t = sum(p_model[i,k,t_RH]*round(z[i,k,t_RH].x,2) for k in K for i in Mk[k,t_RH])
            costo_backorder_dia_t = 0
            
            var_compra = {(i,k,t):round(z[i,k,t].x,2) for t in TT for k in K for i in Mk[k,t]}
            
            for k in K:
                
                inventario[t_RH][0][k] = round(I_0[k], 2)
                
                qp = sum(z[i,k,t_RH].x for i in Mk[k,t_RH]) + I_0[k]
                
                if round(qp,2) < round(d_model[k,t_RH],2):
                    
                    costo_backorder_dia_t+=bo_EV[k]*(d_model[k,t_RH] - qp)
                    compra_extra[t_RH][k] = round(d_model[k,t_RH] - qp,2)
                    inventario[t_RH][1][k] = 0
                    
                else:
                    inventario[t_RH][1][k] = qp - d_model[k,t_RH]
                    
            for k in K:
                for i in Mk[k,t_RH]:
                    if var_compra[i,k,t_RH] > 0:
                        solucionTTP[t_RH][0][i] = True
                        solucionTTP[t_RH][1][i]+= var_compra[i,k,t_RH]
                        solucionTTP[t_RH][2][i][k]=True
                        solucionTTP[t_RH][3][i][k]=var_compra[i,k,t_RH]
                        solucionTTP[t_RH][6][k]+=var_compra[i,k,t_RH]
            
            Rutas_finales, solucionTTP, solucionTTP[t_RH][8]  = Genera_ruta_at_t(solucionTTP, t_RH, max_cij, c, Q, flag_routing)
            
            solucionTTP[t_RH].append(Rutas_finales.copy())
            solucionTTP[t_RH][7] = costo_compra_dia_t
            costo_ruta_dia_t = solucionTTP[t_RH][8] #+ len(Rutas_finales[t_RH].keys())*unload_time_depot
            
            Profit_day_t = costo_revenue_dia_t - costo_compra_dia_t - costo_backorder_dia_t
            obj_day_t = alpha*Profit_day_t-(1-alpha)*costo_ruta_dia_t
                
            final_policy[t_RH]=(solucionTTP[t_RH].copy(), inventario[t_RH].copy(), compra_extra[t_RH], costo_ruta_dia_t, Profit_day_t,  costo_revenue_dia_t, costo_compra_dia_t, costo_backorder_dia_t, obj_day_t)
            
            FO_policy_routing+=costo_ruta_dia_t
            FO_policy_profit+=Profit_day_t
            FO_policy_purchase+=costo_compra_dia_t
            FO_policy_bo+=costo_backorder_dia_t
            FO_policy_revenue+=costo_revenue_dia_t
            FO_policy_obj+=obj_day_t 
            
            for k in K:
                I_0[k] = round((1-per_por[k])*inventario[t_RH][1][k],2)
                          
        else:
            print('Problem infeasible')
            print(t_RH)
            break
        
    tiempoOpti = time.time()-tiempoInicio
    Final_indicators = [FO_policy_obj, FO_policy_routing,  FO_policy_profit, FO_policy_revenue, FO_policy_purchase, FO_policy_bo, tiempoOpti]
        
    return final_policy, Final_indicators

def Stochasctic_Rolling_Horizon_Daganzo(Vertex, V, Products, K, per_por, Periods, T, M, Mk, Q, d, dist_demand_parm, q, dist_q, p, dist_p, c, max_cij, sample, sample_paths, H, gamma, r_EV, bo_EV, alpha, gap, d_log, p_log, q_log, coor, sup_per_veh, k_factor):
    
    tiempoInicio = time.time()
    
    W = range(sample_paths)
    pw = {ww:1/len(W) for ww in W}
    
    BigM = max(coor.values(), key=operator.itemgetter(1))[1]*10
    BigM2 = BigM*2
    BigM3 = BigM2*len(V)
    
    r_bar = np.average([c[0,i]+c[i,0] for i in M])
        
    gammas_RH = {i:round(gamma[i-1],3) for i in M}
    count_i = {i:0 for i in M}
    gamma_i = {i:0 for i in M}
    
    final_policy = {}
    FO_policy_routing = 0
    FO_policy_profit= 0
    FO_policy_purchase = 0
    FO_policy_bo = 0
    FO_policy_revenue = 0
    FO_policy_obj = 0
        
    solucionTTP = {t:[np.zeros(len(V), dtype=bool), np.zeros(len(V), dtype=float), np.zeros((len(V), len(K)), dtype=bool), np.zeros((len(V), len(K)), dtype=float), np.full(len(V) , -1, dtype = int), np.zeros(len(V), dtype=int), np.zeros(len(K), dtype=float), 0, 0] for t in T}
    compra_extra = {t:np.zeros(len(K), dtype = float) for t in T}
    inventario = {t:[[0 for k in K], [0 for k in K]]  for t in range(len(T))}
     
    I_0 = {k:0.0 for k in K} #Initial inventory level   
    dif_routing_cost = []
    dif_heuristic_solution = []
    
    EV_price = (upper_price + lower_price)/2.0
    EV_cap = (upper_quiantity + lower_quiantity)/2.0
    
    for t_RH in T:
        
        if t_RH + H < len(T)-1:
            
            t_max = t_RH+H
        else:
            t_max=len(T)-1
        
        TT = range(t_RH,t_max+1)
        
        '''
        C_MIP = {}
        for i in M:
            for t in TT:
                if t == t_RH:
                    C_MIP[i,t] = (c[0,i]+c[i,0])*gammas_RH[i]
                else:
                    C_MIP[i,t] = (c[0,i]+c[i,0])*gammas_RH[i]
        '''

        d_model = {}
        q_model = {}
        p_model = {}
        
        for k in K:
            for t in TT:
                for ww in W:
                    if t == t_RH:
                        
                        d_model[k,t,ww] = d[k,t,sample]
                            
                        for i in Mk[k,t]:
                            q_model[i,k,t,ww] = q[i,k,t, sample]
                            p_model[i,k,t,ww] = p[i,k,t, sample]
                            
                    else:
                            
                        d_model[k,t,ww] = provides_random_value(0, d_log[k], dist_train)
                        
                        
                        if prices_correlated_time == True or offer_correlated_time == True:
                            
                            l_selected = np.random.randint(0,levels)
                            
                        else:
                            l_selected = -1
                            
                        for i in Mk[k,t]:
                            
                            if prices_correlated_time == True and offer_correlated_time == True:
                                
                                if l_selected == 0:
                                    
                                    l_price = 1
                                    l_offer = 0
                                    
                                elif l_selected == 1:
                                    
                                    l_price = 0
                                    l_offer = 1
                                    
                                elif l_selected == 2:
                                    
                                    l_price = 2
                                    l_offer = 2
                                    
                                p_model[i,k,t,ww] = provides_random_value(1, p_log[i,k], dist_train, l_price, EV_price, 'price')
                                q_model[i,k,t,ww] = provides_random_value(0, q_log[i,k], dist_train, l_offer, EV_cap, 'offer')
                                    
                            else:
                            
                                p_model[i,k,t,ww] = provides_random_value(1, p_log[i,k], dist_train, l_selected, EV_price, 'price')
                                q_model[i,k,t,ww] = provides_random_value(0, q_log[i,k], dist_train, l_selected, EV_cap, 'offer')
                        
                        
                        
        m = gu.Model('Inventory')
        
        #Inventory variables
        z = {(i,k,t,ww):m.addVar(vtype=gu.GRB.CONTINUOUS, name="z_"+str((i,k,t,ww))) for t in TT for k in K for i in Mk[k,t] for ww in W}
        w = {(i,t,ww):m.addVar(vtype=gu.GRB.BINARY, name="w_"+str((i,t,ww))) for t in TT for i in V for ww in W} #ahora indexo en el subconjunto V sin embargo el 0 no se toma en el modelo solo en Daganzo
        ii = {(k,t,ww):m.addVar(vtype=gu.GRB.CONTINUOUS, name="i_"+str((k,t,ww))) for k in K for t in TT for ww in W}
        bo = {(k,t,ww):m.addVar(vtype=gu.GRB.CONTINUOUS, name="bo_"+str((k,t,ww))) for t in TT for k in K for ww in W}
        
        #DAGANZO VARIABLES
        x_min = {(t,ww):m.addVar(vtype=gu.GRB.CONTINUOUS, name="x_min"+str((t,ww))) for t in TT for ww in W}
        x_max = {(t,ww):m.addVar(vtype=gu.GRB.CONTINUOUS, name="x_max"+str((t,ww))) for t in TT for ww in W}
        y_min = {(t,ww):m.addVar(vtype=gu.GRB.CONTINUOUS, name="y_min"+str((t,ww))) for t in TT for ww in W}
        y_max = {(t,ww):m.addVar(vtype=gu.GRB.CONTINUOUS, name="y_max"+str((t,ww))) for t in TT for ww in W}
        
        width = {(t,ww):m.addVar(vtype=gu.GRB.CONTINUOUS, name="width"+str((t,ww))) for t in TT for ww in W}
        height = {(t,ww):m.addVar(vtype=gu.GRB.CONTINUOUS, name="height"+str((t,ww))) for t in TT for ww in W}
        
        Area = {(t,ww):m.addVar(vtype=gu.GRB.CONTINUOUS, name="Area"+str((t,ww))) for t in TT for ww in W}
        AN = {(t,ww):m.addVar(vtype=gu.GRB.CONTINUOUS, name="AN"+str((t,ww))) for t in TT for ww in W}
        #AN2 = {(t,ww):m.addVar(vtype=gu.GRB.CONTINUOUS, name="AN2"+str((t,ww))) for t in TT for ww in W}
        L = {(t,ww):m.addVar(vtype=gu.GRB.CONTINUOUS, name="L"+str((t,ww))) for t in TT for ww in W}
        
        #aux1 = {(i,t,ww):m.addVar(vtype=gu.GRB.CONTINUOUS, name="aux1_"+str((i,t,ww))) for t in TT for i in M for ww in W}
        #aux2 = {(i,t,ww):m.addVar(vtype=gu.GRB.CONTINUOUS, name="aux2_"+str((i,t,ww))) for t in TT for i in M for ww in W} 
        
        for ww in W:
            
            #Daganzo
            for t in TT:
                
                m.addConstr(w[0,t,ww] == 1)
                
                m.addConstr(x_min[t,ww] <= x_max[t,ww])
                m.addConstr(y_min[t,ww] <= y_max[t,ww])

                m.addConstr(width[t,ww] >= x_max[t,ww] - x_min[t,ww])
                m.addConstr(width[t,ww] <= x_max[t,ww] - x_min[t,ww])

                m.addConstr(height[t,ww] >= y_max[t,ww] - y_min[t,ww])
                m.addConstr(height[t,ww] <= y_max[t,ww] - y_min[t,ww])
                
                m.addConstr(Area[t,ww] == width[t,ww]*height[t,ww])
                
                m.addConstr(AN[t,ww] == gu.quicksum(w[i,t,ww]*Area[t,ww] for i in M))
                
                '''
                m.addConstr(AN[t,ww] == gu.quicksum(aux1[i,t,ww] for i in M))
                
                for i in M:
                    m.addConstr(aux1[i,t,ww] <= Area[t,ww])
                    m.addConstr(aux1[i,t,ww] >= Area[t,ww] - BigM2*(1-w[i,t,ww]))
                    m.addConstr(aux1[i,t,ww] <= BigM2*w[i,t,ww])
                    
                    m.addConstr(aux2[i,t,ww] <= AN[t,ww])
                    m.addConstr(aux2[i,t,ww] >= AN[t,ww] - BigM3*(1-w[i,t,ww]))
                    m.addConstr(aux2[i,t,ww] <= BigM3*w[i,t,ww])
                    
                #m.addConstr(L[t,ww]*L[t,ww] == (0.9**2)*AN[t,ww] + ((0.45/sup_per_veh)**2)*gu.quicksum(aux2[i,t,ww] for i in M))
                #m.addGenConstrPow(AN[t,ww], AN2[t,ww], 0.5) #agregar restriccion con potencia de algo 
                '''
                
                #m.addConstr(L[t,ww]*L[t,ww] == (0.9**2)*AN[t,ww] + ((k_factor/sup_per_veh)**2)*gu.quicksum(w[i,t,ww]*AN[t,ww] for i in M))
                
                #m.addConstr(L[t,ww]*L[t,ww] == (0.9**2)*AN[t,ww] + 1.8*(k_factor/sup_per_veh**2)*gu.quicksum(w[i,t,ww]*AN[t,ww] for i in M) +  ((k_factor/sup_per_veh**2)**2)*gu.quicksum(w[i,t,ww]*AN[t,ww] for i in M))
                
                m.addConstr(L[t,ww]*L[t,ww] == (0.81 + ((1.8*k_factor)/sup_per_veh**2) + ((k_factor**2)/(sup_per_veh**4)))*AN[t,ww])
                
                
                for i in V:
                    m.addConstr(x_min[t,ww] <= coor[i][0] + BigM*(1 - w[i,t,ww]))   
                    m.addConstr(coor[i][0] - BigM*(1 - w[i,t,ww]) <= x_max[t,ww])
                    m.addConstr(y_min[t,ww] <= coor[i][1] + BigM*(1 - w[i,t,ww]))
                    m.addConstr(coor[i][1] - BigM*(1 - w[i,t,ww]) <= y_max[t,ww])
                    
            #Modelo normal
            
            for k in K:
                m.addConstr(ii[k,t_RH,ww] == I_0[k] + gu.quicksum(z[i,k,t_RH, ww] for i in Mk[k,t_RH]) - d_model[k,t_RH, ww] + bo[k,t_RH, ww])
            
    
            for k in K:            
                for t in TT:
                    if t > t_RH:
                        m.addConstr(ii[k,t,ww] == (ii[k,t-1,ww]*(1-per_por[k])) + gu.quicksum(z[i,k,t,ww] for i in Mk[k,t]) - d_model[k,t,ww] + bo[k,t,ww])

            for t in TT:
                for k in K:
                    for i in Mk[k,t]:
                        m.addConstr(z[i,k,t,ww] <= q_model[i,k,t,ww]*w[i,t,ww])
                        
            for t in TT:
                for i in M:
                    m.addConstr(gu.quicksum(z[i,k,t,ww] for k in K if (i,k,t,ww) in z) <= Q )
                    m.addConstr((c[0,i]+c[i,0])*w[i,t,ww] <= daily_time )
            
            for k in K:
                
                m.addConstr(bo[k,t_RH,ww] ==  gu.quicksum(bo[k,t_RH,www]/len(W) for www in W))
                
                for i in Mk[k,t_RH]:
                    m.addConstr(z[i,k,t_RH,ww] ==  gu.quicksum(z[i,k,t_RH,www]/len(W) for www in W))
                    
                m.addConstr(ii[k,t_RH,ww] ==  gu.quicksum(ii[k,t_RH,www]/len(W) for www in W))
                     
            for i in M:
                m.addConstr(w[i,t_RH,ww] ==  gu.quicksum(w[i,t_RH,www]/len(W) for www in W))
                
            
            m.addConstr(L[t_RH,ww] ==  gu.quicksum(L[t_RH,www]/len(W) for www in W))
            
        purchase = gu.quicksum(pw[ww]*gu.quicksum(p_model[i,k,t,ww]*z[i,k,t,ww] for t in TT for k in K for i in Mk[k,t]) for ww in W)
        backorders = gu.quicksum(pw[ww]*gu.quicksum(bo_EV[k]*bo[k,t,ww] for k in K for t in TT) for ww in W)
        #route = gu.quicksum(pw[ww]*gu.quicksum(C_MIP[i,t]*w[i,t,ww] for t in TT for i in M) for ww in W)*increase_route
        route = gu.quicksum(pw[ww]*gu.quicksum(L[t,ww] for t in TT) for ww in W)*increase_route
        revenue = gu.quicksum(pw[ww]*gu.quicksum(r_EV[k]*d_model[k,t,ww] for t in TT for k in K) for ww in W)
        
        m.setObjective(((1-alpha)*((-1)*route)) + (alpha)*(revenue-purchase-backorders), sense = gu.GRB.MAXIMIZE)
                
        m.update()
        m.setParam('OutputFlag',0)
        m.setParam('MIPGap',gap)
        m.setParam('TimeLimit',1)
        #m.setParam("FuncMaxVal",10e+20)
        m.setParam("NonConvex",2)
        m.optimize()
        
        if m.Status==2 or m.Status==9:
            
            costo_revenue_dia_t = sum(r_EV[k]*d_model[k,t_RH,0] for k in K)
            costo_compra_dia_t = sum(p_model[i,k,t_RH,0]*round(z[i,k,t_RH,0].x,2) for k in K for i in Mk[k,t_RH])
            costo_backorder_dia_t =  sum(bo_EV[k]*round(bo[k,t_RH,0].x,2) for k in K)
            costo_ruta_dia_t_MIP = L[t_RH,0].x 

            var_compra = {(i,k,t):round(z[i,k,t,0].x,2) for t in TT for k in K for i in Mk[k,t]}
            
            for k in K:
                compra_extra[t_RH][k] = round(bo[k,t_RH,0].x,2)
                inventario[t_RH][0][k] = round(I_0[k], 2)
                inventario[t_RH][1][k] = round(ii[k,t_RH,0].x, 2)
                    
            for k in K:
                for i in Mk[k,t_RH]:
                    if var_compra[i,k,t_RH] > 0:
                        solucionTTP[t_RH][0][i] = True
                        solucionTTP[t_RH][1][i]+= var_compra[i,k,t_RH]
                        solucionTTP[t_RH][2][i][k]=True
                        solucionTTP[t_RH][3][i][k]=var_compra[i,k,t_RH]
                        solucionTTP[t_RH][6][k]+=var_compra[i,k,t_RH]
            '''          
            if flag_dif_routing == True:
                solucionTTP_copia = solucionTTP.copy()
                Rutas_finales_copia, solucionTTP_copia , FO_route_dif  = Genera_ruta_at_t(solucionTTP_copia, t_RH, max_cij, c, Q, heuristic_dif)                
            '''
                        
            Rutas_finales, solucionTTP, solucionTTP[t_RH][8]  = Genera_ruta_at_t(solucionTTP, t_RH, max_cij, c, Q, flag_routing)
            
            solucionTTP[t_RH].append(Rutas_finales.copy())
            solucionTTP[t_RH][7] = costo_compra_dia_t
            costo_ruta_dia_t = solucionTTP[t_RH][8] #+ len(Rutas_finales[t_RH].keys())*unload_time_depot
            
            Profit_day_t = costo_revenue_dia_t - costo_compra_dia_t - costo_backorder_dia_t
            obj_day_t = alpha*Profit_day_t-(1-alpha)*costo_ruta_dia_t
            
            if costo_ruta_dia_t > 0:
                dif_routing_cost.append(np.abs((costo_ruta_dia_t_MIP-costo_ruta_dia_t)/costo_ruta_dia_t*100))
            elif costo_ruta_dia_t_MIP >0:
                dif_routing_cost.append(np.abs((costo_ruta_dia_t-costo_ruta_dia_t_MIP)/costo_ruta_dia_t_MIP*100))
            else:
                dif_routing_cost.append(0)       
            '''    
            if flag_dif_routing == True:
                if solucionTTP[t_RH][8] > 0:
                    dif_heuristic_solution.append(np.abs((FO_route_dif-solucionTTP[t_RH][8])/solucionTTP[t_RH][8]*100))
            '''
            
            final_policy[t_RH]=(solucionTTP[t_RH].copy(), inventario[t_RH].copy(), compra_extra[t_RH], costo_ruta_dia_t, Profit_day_t,  costo_revenue_dia_t, costo_compra_dia_t, costo_backorder_dia_t,obj_day_t, costo_ruta_dia_t_MIP)
            
            FO_policy_routing+=costo_ruta_dia_t
            FO_policy_profit+=Profit_day_t
            FO_policy_purchase+=costo_compra_dia_t
            FO_policy_bo+=costo_backorder_dia_t
            FO_policy_revenue+=costo_revenue_dia_t
            FO_policy_obj+=obj_day_t 
            
            for k in K:
                I_0[k] = round((1-per_por[k])*ii[k,t_RH,0].x,2)
            
        else:
            print('Problem infeasible')
            print(t_RH)
            break
        
        
        if T_aux == True:
            break
    
    tiempoOpti = time.time()-tiempoInicio
    Final_indicators = [FO_policy_obj, FO_policy_routing,  FO_policy_profit, FO_policy_revenue, FO_policy_purchase, FO_policy_bo, tiempoOpti, round(np.average(dif_routing_cost),2), round(m.MIPGap,2)]
    
    for i in M:
        #gamma_node = gammas_RH[i]
        gamma_in_route = [gammas_RH[i]]
        #gamma_in_route = []
        direct = (c[0,i] + c[i,0])
        
        for t in T:
            if final_policy[t][0][4][i] != -1:
                index_ruta = final_policy[t][0][4][i]
                time_route = final_policy[t][0][9][t][index_ruta][2]
                sup_per_rou = len(final_policy[t][0][9][t][index_ruta][3])
                avg_time = time_route/sup_per_rou 
                gamma_in_route.append(round((avg_time/direct),2))
                
                if len(gamma_in_route) >1:
                    count_i[i]=1   
                    
            if T_aux == True:
                break
                    
        gamma_i[i] = round(np.average(gamma_in_route),2)         
        
    return final_policy, Final_indicators, count_i, gamma_i

def Stochasctic_Rolling_Horizon_xij(Vertex, V, Products, K, per_por, Periods, T, M, Mk, Q, d, dist_demand_parm, q, dist_q, p, dist_p, c, max_cij, sample, sample_paths, H, gamma, r_EV, bo_EV, alpha, gap, d_log, p_log, q_log, Km):
    
    tiempoInicio = time.time()
    
    W = range(sample_paths)
    F = range(len(M))
    
    pw = {ww:1/len(W) for ww in W}
        
    gammas_RH = {i:round(gamma[i-1],3) for i in M}
    count_i = {i:0 for i in M}
    gamma_i = {i:0 for i in M}
    
    final_policy = {}
    FO_policy_routing = 0
    FO_policy_profit= 0
    FO_policy_purchase = 0
    FO_policy_bo = 0
    FO_policy_revenue = 0
    FO_policy_obj = 0
        
    solucionTTP = {t:[np.zeros(len(V), dtype=bool), np.zeros(len(V), dtype=float), np.zeros((len(V), len(K)), dtype=bool), np.zeros((len(V), len(K)), dtype=float), np.full(len(V) , -1, dtype = int), np.zeros(len(V), dtype=int), np.zeros(len(K), dtype=float), 0, 0] for t in T}
    compra_extra = {t:np.zeros(len(K), dtype = float) for t in T}
    inventario = {t:[[0 for k in K], [0 for k in K]]  for t in range(len(T))}
     
    I_0 = {k:0.0 for k in K} #Initial inventory level 
    
    dif_routing_cost = []
    
    EV_price = (upper_price + lower_price)/2.0
    EV_cap = (upper_quiantity + lower_quiantity)/2.0
    
    for t_RH in T:
        
        if t_RH + H < len(T)-1:
            
            t_max = t_RH+H
        else:
            t_max=len(T)-1
        
        TT = range(t_RH,t_max+1)
        '''
        C_MIP = {}
        for i in M:
            for t in TT:
                if t == t_RH:
                    C_MIP[i,t] = (c[0,i]+c[i,0])*gammas_RH[i]
                else:
                    C_MIP[i,t] = (c[0,i]+c[i,0])*gammas_RH[i]
        '''
                    
        d_model = {}
        q_model = {}
        p_model = {}
        
        for k in K:
            for t in TT:
                for ww in W:
                    if t == t_RH:
                        
                        d_model[k,t,ww] = d[k,t,sample]
                            
                        for i in Mk[k,t]:
                            q_model[i,k,t,ww] = q[i,k,t, sample]
                            p_model[i,k,t,ww] = p[i,k,t, sample]
                            
                    else:
                            
                        d_model[k,t,ww] = provides_random_value(0, d_log[k], dist_train)
                        
                        
                        if prices_correlated_time == True or offer_correlated_time == True:
                            
                            l_selected = np.random.randint(0,levels)
                            
                        else:
                            l_selected = -1
                            
                        for i in Mk[k,t]:
                            
                            if prices_correlated_time == True and offer_correlated_time == True:
                                
                                if l_selected == 0:
                                    
                                    l_price = 1
                                    l_offer = 0
                                    
                                elif l_selected == 1:
                                    
                                    l_price = 0
                                    l_offer = 1
                                    
                                elif l_selected == 2:
                                    
                                    l_price = 2
                                    l_offer = 2
                                    
                                p_model[i,k,t,ww] = provides_random_value(1, p_log[i,k], dist_train, l_price, EV_price, 'price')
                                q_model[i,k,t,ww] = provides_random_value(0, q_log[i,k], dist_train, l_offer, EV_cap, 'offer')
                                    
                            else:
                            
                                p_model[i,k,t,ww] = provides_random_value(1, p_log[i,k], dist_train, l_selected, EV_price, 'price')
                                q_model[i,k,t,ww] = provides_random_value(0, q_log[i,k], dist_train, l_selected, EV_cap, 'offer')
                            
        c_ = {(i,j,v,t,ww):0 for (i,j) in c.keys() for v in F for t in TT for ww in W}
        
        m = gu.Model('Inventory')
        
        #Decisions variables
        x = m.addVars(c_.keys(), vtype=gu.GRB.BINARY, name="x_")
        z = {(i,k,v,t,ww):m.addVar(vtype=gu.GRB.CONTINUOUS, name="z_"+str((i,k,v,t,ww))) for v in F for t in TT for k in K for i in Mk[k,t] for ww in W}
        w = {(i,v,t,ww):m.addVar(vtype=gu.GRB.BINARY, name="w_"+str((i,k,t,ww))) for v in F for t in TT for i in M for ww in W}
        ii = {(k,t,ww):m.addVar(vtype=gu.GRB.CONTINUOUS, name="i_"+str((k,t,ww))) for k in K for t in TT for ww in W}
        bo = {(k,t,ww):m.addVar(vtype=gu.GRB.CONTINUOUS, name="bo_"+str((k,t,ww))) for t in TT for k in K for ww in W}
        u = {(i,v,t,ww):m.addVar(vtype=gu.GRB.CONTINUOUS, name="u_"+str((i,v,t,ww))) for v in F for i in M for t in TT for ww in W}
        
        
        for ww in W:
            
            for k in K:
                m.addConstr(ii[k,t_RH,ww] == I_0[k] + gu.quicksum(z[i,k,v,t_RH, ww] for v in F for i in Mk[k,t_RH]) - d_model[k,t_RH, ww] + bo[k,t_RH, ww])
            
    
            for k in K:            
                for t in TT:
                    if t > t_RH:
                        m.addConstr(ii[k,t,ww] == (ii[k,t-1,ww]*(1-per_por[k])) + gu.quicksum(z[i,k,v,t,ww] for v in F for i in Mk[k,t]) - d_model[k,t,ww] + bo[k,t,ww])

            for t in TT:
                for k in K:
                    for i in Mk[k,t]:
                        for v in F:    
                            m.addConstr(z[i,k,v,t,ww] <= q_model[i,k,t,ww]*w[i,v,t,ww])
            '''                
            for t in TT:
                for i in M:
                    for v in F:
                        m.addConstr(w[i,v,t,ww] <= gu.quicksum(z[i,k,v,t,ww]/q_model[i,k,t,ww] for k in Km[i]))
            '''                
            
            for t in TT:
                for v in F:
                    m.addConstr(gu.quicksum(z[i,k,v,t,ww] for i in M for k in Km[i]) <= Q)
            
            
            for t in TT:
                for i in M:
                    m.addConstr(gu.quicksum(w[i,v,t,ww] for v in F ) <= 1)
            
                        
            for t in TT:
                for i in M:
                    for v in F:    
                        m.addConstr(gu.quicksum(z[i,k,v,t,ww] for k in Km[i]) <= Q*w[i,v,t,ww])
                        
                        
            for t in TT:
                for v in F:
                    m.addConstr(gu.quicksum(x[0,j,v,t,ww] for j in M ) <= 1)
            
            for t in TT:
                for v in F:
                    for hh in M:
                        m.addConstr(x.sum(hh,'*',v,t,ww) ==  w[hh,v,t,ww])
                        m.addConstr(x.sum('*',hh,v,t,ww) ==  w[hh,v,t,ww])
                    
            for t in TT:
                for v in F:
                    m.addConstr(gu.quicksum(c[i,j]*x[i,j,v,t,ww] for (i,j) in c.keys()) <= daily_time)
                    
            for t in TT:
                for i in M:
                    for v in F:
                        m.addConstr(w[i,v,t,ww] <= gu.quicksum(x[0,j,v,t,ww] for j in M))
                        
                        
            for t in TT:
                for i in M:
                    for v in F:
                        for j in M:
                            if i!=j:
                                m.addConstr(u[i,v,t,ww] - u[j,v,t,ww] + len(M)*x[i,j,v,t,ww] <= len(M)-1 )
                
            for t in TT:
                for v in F:
                    if v > 0:
                        m.addConstr(gu.quicksum(x[0, j,v,t,ww]  for j in M) <= gu.quicksum(x[0, j, v-1,t,ww]  for j in M))
                        
                        for k in K:    
                            m.addConstr(gu.quicksum(z[i,k,v,t,ww]  for i in Mk[k,t]) <= gu.quicksum(z[i,k,v-1,t,ww]  for i in Mk[k,t]))
            
            
            for k in K:
                
                m.addConstr(bo[k,t_RH,ww] ==  gu.quicksum(bo[k,t_RH,www]/len(W) for www in W))
                
                for i in Mk[k,t_RH]:
                    for v in F:    
                        m.addConstr(z[i,k,v,t_RH,ww] ==  gu.quicksum(z[i,k,v,t_RH,www]/len(W) for www in W))
                    
                m.addConstr(ii[k,t_RH,ww] ==  gu.quicksum(ii[k,t_RH,www]/len(W) for www in W))
                     
            for i in M:
                for v in F:    
                    m.addConstr(w[i,v,t_RH,ww] ==  gu.quicksum(w[i,v,t_RH,www]/len(W) for www in W))
                    
            for i in V:
                for j in V:
                    if i!=j:
                        for v in F:
                            m.addConstr(x[i, j,v,t_RH,ww] ==  gu.quicksum(x[i,j,v,t_RH,www]/len(W) for www in W))
                            
            
        purchase = gu.quicksum(pw[ww]*gu.quicksum(p_model[i,k,t,ww]*z[i,k,v,t,ww] for v in F for t in TT for k in K for i in Mk[k,t]) for ww in W)
        backorders = gu.quicksum(pw[ww]*gu.quicksum(bo_EV[k]*bo[k,t,ww] for k in K for t in TT) for ww in W)
        #route = gu.quicksum(pw[ww]*gu.quicksum(C_MIP[i,t]*w[i,t,ww] for t in TT for i in M) for ww in W)*increase_route
        
        route = gu.quicksum(pw[ww]*gu.quicksum(c[i,j]*x[i,j,v,t,ww] for v in F for t in TT for (i,j) in c.keys()) for ww in W)*increase_route
        revenue = gu.quicksum(pw[ww]*gu.quicksum(r_EV[k]*d_model[k,t,ww] for t in TT for k in K) for ww in W)
        
        m.setObjective(((1-alpha)*((-1)*route)) + (alpha)*(revenue-purchase-backorders), sense = gu.GRB.MAXIMIZE)
        
       
        m.update()
        m.setParam('OutputFlag',1)
        m.setParam('MIPGap',gap)
        m.setParam('TimeLimit', 3600)
        m.optimize()
        
        if m.Status==2 or m.Status==9:
            
            costo_revenue_dia_t = sum(r_EV[k]*d_model[k,t_RH,0] for k in K)
            costo_compra_dia_t = sum(p_model[i,k,t_RH,0]*round(z[i,k,v,t_RH,0].x,2) for v in F for k in K for i in Mk[k,t_RH])
            costo_backorder_dia_t =  sum(bo_EV[k]*round(bo[k,t_RH,0].x,2) for k in K)
            costo_ruta_dia_t_MIP = sum(c[i,j]*x[i,j,v,t_RH,0].x for v in F for (i,j) in c.keys())
            
            var_compra = {(i,k,t):round(sum(z[i,k,v,t,0].x for v in F),2) for t in TT for k in K for i in Mk[k,t]}
            
            for k in K:
                compra_extra[t_RH][k] = round(bo[k,t_RH,0].x,2)
                inventario[t_RH][0][k] = round(I_0[k], 2)
                inventario[t_RH][1][k] = round(ii[k,t_RH,0].x, 2)
                    
            for k in K:
                for i in Mk[k,t_RH]:
                    if var_compra[i,k,t_RH] > 0:
                        solucionTTP[t_RH][0][i] = True
                        solucionTTP[t_RH][1][i]+= var_compra[i,k,t_RH]
                        solucionTTP[t_RH][2][i][k]=True
                        solucionTTP[t_RH][3][i][k]=var_compra[i,k,t_RH]
                        solucionTTP[t_RH][6][k]+=var_compra[i,k,t_RH]
                        
            Rutas_finales, solucionTTP, solucionTTP[t_RH][8]  = Genera_ruta_at_t(solucionTTP, t_RH, max_cij, c, Q, flag_routing)
            
            solucionTTP[t_RH].append(Rutas_finales.copy())
            solucionTTP[t_RH][7] = costo_compra_dia_t
            costo_ruta_dia_t = solucionTTP[t_RH][8] #+ len(Rutas_finales[t_RH].keys())*unload_time_depot
            
            Profit_day_t = costo_revenue_dia_t - costo_compra_dia_t - costo_backorder_dia_t
            obj_day_t = alpha*Profit_day_t-(1-alpha)*costo_ruta_dia_t
            
            if costo_ruta_dia_t > 0:
                dif_routing_cost.append(np.abs((costo_ruta_dia_t_MIP-costo_ruta_dia_t)/costo_ruta_dia_t*100))
            elif costo_ruta_dia_t_MIP >0:
                dif_routing_cost.append(np.abs((costo_ruta_dia_t-costo_ruta_dia_t_MIP)/costo_ruta_dia_t_MIP*100))
            else:
                dif_routing_cost.append(0)                
            
            final_policy[t_RH]=(solucionTTP[t_RH].copy(), inventario[t_RH].copy(), compra_extra[t_RH], costo_ruta_dia_t, Profit_day_t,  costo_revenue_dia_t, costo_compra_dia_t, costo_backorder_dia_t,obj_day_t, costo_ruta_dia_t_MIP)
            
            FO_policy_routing+=costo_ruta_dia_t
            FO_policy_profit+=Profit_day_t
            FO_policy_purchase+=costo_compra_dia_t
            FO_policy_bo+=costo_backorder_dia_t
            FO_policy_revenue+=costo_revenue_dia_t
            FO_policy_obj+=obj_day_t 
            
            for k in K:
                I_0[k] = round((1-per_por[k])*ii[k,t_RH,0].x,2)
                          
        else:
            print('Problem infeasible')
            print(t_RH)
            break
        
        if T_aux == True:
            break
    
    tiempoOpti = time.time()-tiempoInicio
    Final_indicators = [FO_policy_obj, FO_policy_routing,  FO_policy_profit, FO_policy_revenue, FO_policy_purchase, FO_policy_bo, tiempoOpti, round(np.average(dif_routing_cost),2), round(m.MIPGap,2)]
    
    for i in M:
        #gamma_node = gammas_RH[i]
        gamma_in_route = [gammas_RH[i]]
        #gamma_in_route = []
        direct = (c[0,i] + c[i,0])
        
        for t in T:
            if final_policy[t][0][4][i] != -1:
                index_ruta = final_policy[t][0][4][i]
                time_route = final_policy[t][0][9][t][index_ruta][2]
                sup_per_rou = len(final_policy[t][0][9][t][index_ruta][3])
                avg_time = time_route/sup_per_rou 
                gamma_in_route.append(round((avg_time/direct),2))
                
                if len(gamma_in_route) >1:
                    count_i[i]=1 
                    
            if T_aux == True:
                break
                    
        gamma_i[i] = round(np.average(gamma_in_route),2)         
        
    return final_policy, Final_indicators, count_i, gamma_i

def Stochasctic_Rolling_Horizon(Vertex, V, Products, K, per_por, Periods, T, M, Mk, Q, d, dist_demand_parm, q, dist_q, p, dist_p, c, max_cij, sample, sample_paths, H, gamma, r_EV, bo_EV, alpha, gap, d_log, p_log, q_log):
    
    tiempoInicio = time.time()
    
    W = range(sample_paths)
    pw = {ww:1/len(W) for ww in W}
        
    gammas_RH = {i:round(gamma[i-1],3) for i in M}
    count_i = {i:0 for i in M}
    gamma_i = {i:0 for i in M}
    
    final_policy = {}
    FO_policy_routing = 0
    FO_policy_profit= 0
    FO_policy_purchase = 0
    FO_policy_bo = 0
    FO_policy_revenue = 0
    FO_policy_obj = 0
        
    solucionTTP = {t:[np.zeros(len(V), dtype=bool), np.zeros(len(V), dtype=float), np.zeros((len(V), len(K)), dtype=bool), np.zeros((len(V), len(K)), dtype=float), np.full(len(V) , -1, dtype = int), np.zeros(len(V), dtype=int), np.zeros(len(K), dtype=float), 0, 0] for t in T}
    compra_extra = {t:np.zeros(len(K), dtype = float) for t in T}
    inventario = {t:[[0 for k in K], [0 for k in K]]  for t in range(len(T))}
     
    I_0 = {k:0.0 for k in K} #Initial inventory level   
    dif_routing_cost = []
    dif_heuristic_solution = []
    
    EV_price = (upper_price + lower_price)/2.0
    EV_cap = (upper_quiantity + lower_quiantity)/2.0
    
    for t_RH in T:
        
        if t_RH + H < len(T)-1:
            
            t_max = t_RH+H
        else:
            t_max=len(T)-1
        
        TT = range(t_RH,t_max+1)
        
        C_MIP = {}
        for i in M:
            for t in TT:
                if t == t_RH:
                    C_MIP[i,t] = (c[0,i]+c[i,0])*gammas_RH[i]
                else:
                    C_MIP[i,t] = (c[0,i]+c[i,0])*gammas_RH[i]
                    
        d_model = {}
        q_model = {}
        p_model = {}
        
        for k in K:
            for t in TT:
                for ww in W:
                    if t == t_RH:
                        
                        d_model[k,t,ww] = d[k,t,sample]
                            
                        for i in Mk[k,t]:
                            q_model[i,k,t,ww] = q[i,k,t, sample]
                            p_model[i,k,t,ww] = p[i,k,t, sample]
                            
                    else:
                            
                        d_model[k,t,ww] = provides_random_value(0, d_log[k], dist_train)
                        
                        
                        if prices_correlated_time == True or offer_correlated_time == True:
                            
                            l_selected = np.random.randint(0,levels)
                            
                        else:
                            l_selected = -1
                            
                        for i in Mk[k,t]:
                            
                            if prices_correlated_time == True and offer_correlated_time == True:
                                
                                if l_selected == 0:
                                    
                                    l_price = 1
                                    l_offer = 0
                                    
                                elif l_selected == 1:
                                    
                                    l_price = 0
                                    l_offer = 1
                                    
                                elif l_selected == 2:
                                    
                                    l_price = 2
                                    l_offer = 2
                                    
                                p_model[i,k,t,ww] = provides_random_value(1, p_log[i,k], dist_train, l_price, EV_price, 'price')
                                q_model[i,k,t,ww] = provides_random_value(0, q_log[i,k], dist_train, l_offer, EV_cap, 'offer')
                                    
                            else:
                            
                                p_model[i,k,t,ww] = provides_random_value(1, p_log[i,k], dist_train, l_selected, EV_price, 'price')
                                q_model[i,k,t,ww] = provides_random_value(0, q_log[i,k], dist_train, l_selected, EV_cap, 'offer')
                        
                        
                        
        m = gu.Model('Inventory')
        
        #Inventory variables
        z = {(i,k,t,ww):m.addVar(vtype=gu.GRB.CONTINUOUS, name="z_"+str((i,k,t,ww))) for t in TT for k in K for i in Mk[k,t] for ww in W}
        w = {(i,t,ww):m.addVar(vtype=gu.GRB.BINARY, name="w_"+str((i,t,ww))) for t in TT for i in M for ww in W}
        ii = {(k,t,ww):m.addVar(vtype=gu.GRB.CONTINUOUS, name="i_"+str((k,t,ww))) for k in K for t in TT for ww in W}
        bo = {(k,t,ww):m.addVar(vtype=gu.GRB.CONTINUOUS, name="bo_"+str((k,t,ww))) for t in TT for k in K for ww in W}
        
        for ww in W:
            
            for k in K:
                m.addConstr(ii[k,t_RH,ww] == I_0[k] + gu.quicksum(z[i,k,t_RH, ww] for i in Mk[k,t_RH]) - d_model[k,t_RH, ww] + bo[k,t_RH, ww])

    
            for k in K:            
                for t in TT:
                    if t > t_RH:
                        m.addConstr(ii[k,t,ww] == (ii[k,t-1,ww]*(1-per_por[k])) + gu.quicksum(z[i,k,t,ww] for i in Mk[k,t]) - d_model[k,t,ww] + bo[k,t,ww])

            for t in TT:
                for k in K:
                    for i in Mk[k,t]:
                        m.addConstr(z[i,k,t,ww] <= q_model[i,k,t,ww]*w[i,t,ww])
                        
            for t in TT:
                for i in M:
                    m.addConstr(gu.quicksum(z[i,k,t,ww] for k in K if (i,k,t,ww) in z) <= Q )
                    m.addConstr((c[0,i]+c[i,0])*w[i,t,ww] <= daily_time )
            
            for k in K:
                
                m.addConstr(bo[k,t_RH,ww] ==  gu.quicksum(bo[k,t_RH,www]/len(W) for www in W))
                
                for i in Mk[k,t_RH]:
                    m.addConstr(z[i,k,t_RH,ww] ==  gu.quicksum(z[i,k,t_RH,www]/len(W) for www in W))
                    
                m.addConstr(ii[k,t_RH,ww] ==  gu.quicksum(ii[k,t_RH,www]/len(W) for www in W))
                     
            for i in M:
                m.addConstr(w[i,t_RH,ww] ==  gu.quicksum(w[i,t_RH,www]/len(W) for www in W))
            
        purchase = gu.quicksum(pw[ww]*gu.quicksum(p_model[i,k,t,ww]*z[i,k,t,ww] for t in TT for k in K for i in Mk[k,t]) for ww in W)
        backorders = gu.quicksum(pw[ww]*gu.quicksum(bo_EV[k]*bo[k,t,ww] for k in K for t in TT) for ww in W)
        route = gu.quicksum(pw[ww]*gu.quicksum(C_MIP[i,t]*w[i,t,ww] for t in TT for i in M) for ww in W)*increase_route
        revenue = gu.quicksum(pw[ww]*gu.quicksum(r_EV[k]*d_model[k,t,ww] for t in TT for k in K) for ww in W)
        
        m.setObjective(((1-alpha)*((-1)*route)) + (alpha)*(revenue-purchase-backorders), sense = gu.GRB.MAXIMIZE)
                
        m.update()
        m.setParam('OutputFlag',0)
        m.setParam('MIPGap',gap)
        m.optimize()
        
        if m.Status==2 or m.Status==9:
            
            costo_revenue_dia_t = sum(r_EV[k]*d_model[k,t_RH,0] for k in K)
            costo_compra_dia_t = sum(p_model[i,k,t_RH,0]*round(z[i,k,t_RH,0].x,2) for k in K for i in Mk[k,t_RH])
            costo_backorder_dia_t =  sum(bo_EV[k]*round(bo[k,t_RH,0].x,2) for k in K)
            costo_ruta_dia_t_MIP = sum(C_MIP[i,t_RH]*w[i,t_RH,0].x for i in M)
            
            var_compra = {(i,k,t):round(z[i,k,t,0].x,2) for t in TT for k in K for i in Mk[k,t]}
            
            for k in K:
                compra_extra[t_RH][k] = round(bo[k,t_RH,0].x,2)
                inventario[t_RH][0][k] = round(I_0[k], 2)
                inventario[t_RH][1][k] = round(ii[k,t_RH,0].x, 2)
                    
            for k in K:
                for i in Mk[k,t_RH]:
                    if var_compra[i,k,t_RH] > 0:
                        solucionTTP[t_RH][0][i] = True
                        solucionTTP[t_RH][1][i]+= var_compra[i,k,t_RH]
                        solucionTTP[t_RH][2][i][k]=True
                        solucionTTP[t_RH][3][i][k]=var_compra[i,k,t_RH]
                        solucionTTP[t_RH][6][k]+=var_compra[i,k,t_RH]
                        
            if flag_dif_routing == True:
                solucionTTP_copia = solucionTTP.copy()
                Rutas_finales_copia, solucionTTP_copia , FO_route_dif  = Genera_ruta_at_t(solucionTTP_copia, t_RH, max_cij, c, Q, heuristic_dif)                
                        
            Rutas_finales, solucionTTP, solucionTTP[t_RH][8]  = Genera_ruta_at_t(solucionTTP, t_RH, max_cij, c, Q, flag_routing)
            
            solucionTTP[t_RH].append(Rutas_finales.copy())
            solucionTTP[t_RH][7] = costo_compra_dia_t
            costo_ruta_dia_t = solucionTTP[t_RH][8] #+ len(Rutas_finales[t_RH].keys())*unload_time_depot
            
            Profit_day_t = costo_revenue_dia_t - costo_compra_dia_t - costo_backorder_dia_t
            obj_day_t = alpha*Profit_day_t-(1-alpha)*costo_ruta_dia_t
            
            if costo_ruta_dia_t > 0:
                dif_routing_cost.append(np.abs((costo_ruta_dia_t_MIP-costo_ruta_dia_t)/costo_ruta_dia_t*100))
            elif costo_ruta_dia_t_MIP >0:
                dif_routing_cost.append(np.abs((costo_ruta_dia_t-costo_ruta_dia_t_MIP)/costo_ruta_dia_t_MIP*100))
            else:
                dif_routing_cost.append(0)    
                
            if flag_dif_routing == True:
                if solucionTTP[t_RH][8] > 0:
                    dif_heuristic_solution.append(np.abs((FO_route_dif-solucionTTP[t_RH][8])/solucionTTP[t_RH][8]*100))
            else:
                dif_heuristic_solution.append(0)
                
            
            final_policy[t_RH]=(solucionTTP[t_RH].copy(), inventario[t_RH].copy(), compra_extra[t_RH], costo_ruta_dia_t, Profit_day_t,  costo_revenue_dia_t, costo_compra_dia_t, costo_backorder_dia_t,obj_day_t, costo_ruta_dia_t_MIP)
            
            FO_policy_routing+=costo_ruta_dia_t
            FO_policy_profit+=Profit_day_t
            FO_policy_purchase+=costo_compra_dia_t
            FO_policy_bo+=costo_backorder_dia_t
            FO_policy_revenue+=costo_revenue_dia_t
            FO_policy_obj+=obj_day_t 
            
            for k in K:
                I_0[k] = round((1-per_por[k])*ii[k,t_RH,0].x,2)
                          
        else:
            print('Problem infeasible')
            print(t_RH)
            break
        
        
        if T_aux == True:
            break
    
    tiempoOpti = time.time()-tiempoInicio
    Final_indicators = [FO_policy_obj, FO_policy_routing,  FO_policy_profit, FO_policy_revenue, FO_policy_purchase, FO_policy_bo, tiempoOpti, round(np.average(dif_routing_cost),2), round(m.MIPGap,2), dif_heuristic_solution]
    
    for i in M:
        #gamma_node = gammas_RH[i]
        gamma_in_route = [gammas_RH[i]]
        #gamma_in_route = []
        direct = (c[0,i] + c[i,0])
        
        for t in T:
            if final_policy[t][0][4][i] != -1:
                index_ruta = final_policy[t][0][4][i]
                time_route = final_policy[t][0][9][t][index_ruta][2]
                sup_per_rou = len(final_policy[t][0][9][t][index_ruta][3])
                avg_time = time_route/sup_per_rou 
                gamma_in_route.append(round((avg_time/direct),2))
                
                if len(gamma_in_route) >1:
                    count_i[i]=1   
                    
            if T_aux == True:
                break
                    
        gamma_i[i] = round(np.average(gamma_in_route),2)         
        
    return final_policy, Final_indicators, count_i, gamma_i

def tunning_C_daganzo(lower_sample, upper_sample, nombre_archivo, V, K, per_por, T, M, coor, Mk, Q, d, d_EV, q, q_EV, p, p_EV, c, max_cij, dist_demand_parm, r_EV, bo_EV, Km, dist_p, dist_q, best_gamma, d_log, p_log, q_log):
    upper = upper_sample #100   
    lower = lower_sample
    
    gamma = [best_gamma for i in M]
    C_daganzo = [1, 2, 3, 4, 5, 6]
    k_daganzo = [0.35, 0.5, 0.65, 0.8, 0.95]
    
    best_pair = {}
    
    stocha_obj_MY = []
    H = 3
    for C_val in C_daganzo:
        for k_val in k_daganzo:
            
            print(C_val, k_val)

            R_MY = []
    
            for sample in range(lower, upper):
                final_policy, indicators, count_i, gamma_i = Stochasctic_Rolling_Horizon_Daganzo(Vertex, V, Products, K, per_por, Periods, T, M, Mk, Q, d, dist_demand_parm, q, dist_q, p, dist_p, c, max_cij, sample, sample_paths, H, gamma, r_EV, bo_EV, alpha, MIP_gap, d_log, p_log, q_log, coor, C_val, k_val)
                R_MY.append(indicators[0])
            
            best_pair[C_val, k_val] = np.average(R_MY)
        
    with open(nombre_archivo+'.pickle', 'wb') as handle:
        pickle.dump(best_pair, handle, protocol=pickle.HIGHEST_PROTOCOL)

def H_size_and_gamma_performance(lower_sample, upper_sample, lower_H, upper_H, nombre_archivo, V, K, per_por, T, M, coor, Mk, Q, d, d_EV, q, q_EV, p, p_EV, c, max_cij, dist_demand_parm, r_EV, bo_EV, Km, dist_p, dist_q, d_log, p_log, q_log):
    
    upper = upper_sample #100   
    lower = lower_sample #70
    
    gammas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    
    stocha_obj_MY = []
    stocha_obj_DT = []
    stocha_obj_ST = []
    
    for gammaa in gammas:
        print(gammaa)
        gamma = [gammaa for i in M]
        
        R_MY = []
        R_MY_time = []
        H_samples_DT = np.zeros((upper-lower, (upper_H-lower_H)))
        H_samples_ST = np.zeros((upper-lower, (upper_H-lower_H)))            
        H_samples_DT_time = np.zeros((upper-lower, (upper_H-lower_H)))
        H_samples_ST_time = np.zeros((upper-lower, (upper_H-lower_H)))
                
        for sample in range(lower, upper):
            
            final_policy_MY, indicators_MY = Deterministic_Rolling_Horizon(Vertex, V, Products, K, per_por, Periods, T, M, Mk, Q, d, d_EV,q, q_EV,p, p_EV, c, max_cij, sample, 0, gamma, r_EV, bo_EV, alpha, Km, 0, MIP_gap, False)
            R_MY.append(indicators_MY[0])
            R_MY_time.append(indicators_MY[6])
            
            for H in range(lower_H, upper_H):

                final_policy_DT, indicators_DT = Deterministic_Rolling_Horizon(Vertex, V, Products, K, per_por, Periods, T, M, Mk, Q, d, d_EV,q, q_EV,p, p_EV, c, max_cij, sample, H, gamma, r_EV, bo_EV, alpha, Km, 0, MIP_gap, False)
                final_policy_RS, indicators_RS, count_i_RS, gamma_i_RS = Stochasctic_Rolling_Horizon(Vertex, V, Products, K, per_por, Periods, T, M, Mk, Q, d, dist_demand_parm, q, dist_q, p, dist_p, c, max_cij, sample, sample_paths, H, gamma, r_EV, bo_EV, alpha, MIP_gap, d_log, p_log, q_log)
                
                H_samples_DT[sample-lower, H-lower_H] = indicators_DT[0]
                H_samples_ST[sample-lower, H-lower_H] = indicators_RS[0]
                
                H_samples_DT_time[sample-lower, H-lower_H] = indicators_DT[6]
                H_samples_ST_time[sample-lower, H-lower_H] = indicators_RS[6]
        
        stocha_obj_MY.append((np.average(R_MY), np.average(R_MY_time)))    
        stocha_obj_DT.append((np.average(H_samples_DT, axis = 0), np.average(H_samples_DT_time, axis = 0)))
        stocha_obj_ST.append((np.average(H_samples_ST, axis = 0), np.average(H_samples_ST_time, axis = 0)))
                     
    r_Hs = [stocha_obj_MY, stocha_obj_DT, stocha_obj_ST]
    
    with open(nombre_archivo+'.pickle', 'wb') as handle:
        pickle.dump(r_Hs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def Gap_No_samplepaths_performance(lower_sample, upper_sample, nombre_archivo):
    upper = upper_sample #100  
    lower = lower_sample #120
    
    gaps_values = [0.01, 0.02, 0.03, 0.04, 0.05]
    No_sample_paths = [3, 5, 10, 15, 20]
    
    results = {}
    gamma = [0.6 for i in M]
    H = 3
    
    for gap in gaps_values:
        print(gap)
        for sample_paths in No_sample_paths:
            print(sample_paths)
            R_ST = []
            R_ST_time = []
            for sample in range(lower, upper):
                final_policy_RS, indicators_RS, count_i_RS, gamma_i_RS = Stochasctic_Rolling_Horizon(Vertex, V, Products, K, per_por, Periods, T, M, Mk, Q, d, dist_demand_parm, q, dist_q, p, dist_p, c, max_cij, sample, sample_paths, H, gamma, r_EV, bo_EV, alpha, gap)
                R_ST.append(indicators_RS[0])
                R_ST_time.append(indicators_RS[6])
                
            results[gap, sample_paths] = ((np.average(R_ST), np.average(R_ST_time)))
            
    with open(nombre_archivo+'.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def gamma_distance_performance(lower_sample, upper_sample, H, nombre_archivo, V, K, per_por, T, M, coor, Mk, Q, d, d_EV, q, q_EV, p, p_EV, c, max_cij, dist_demand_parm, r_EV, bo_EV, Km, dist_p, dist_q, d_log, p_log, q_log):
    gammas_dist = {}
    upper = upper_sample #20
    lower = lower_sample #0
    
    tiempos =[15, 30, 45, 60, 75, 90, 105, 120, 135, 160]
    
    for ti in tiempos:
        for i in M:
            cont = 0
            for j in M:
                if i != j:
                    if c[i,j] <= ti:
                        cont+=1
                        
            gammas_dist[i,ti] = 1/(1+cont)
    
    obj_history_time = []
    for ti in tiempos:
        print(ti)
        gamma = [gammas_dist[i, ti] for i in M]
        
        R_ST = []
        
        for sample in range(lower, upper):
            
            final_policy_RS, indicators_RS, count_i_RS, gamma_i_RS = Stochasctic_Rolling_Horizon(Vertex, V, Products, K, per_por, Periods, T, M, Mk, Q, d, dist_demand_parm, q, dist_q, p, dist_p, c, max_cij, sample, sample_paths, H, gamma, r_EV, bo_EV, alpha, MIP_gap, d_log, p_log, q_log)
            R_ST.append(indicators_RS[0])
            
        obj_history_time.append(round(np.average(R_ST),2))

    Obj_times_summary = {0: [tiempos, obj_history_time]}
    
    with open(nombre_archivo+'.pickle', 'wb') as handle:
        pickle.dump(Obj_times_summary, handle, protocol=pickle.HIGHEST_PROTOCOL)        
        
def Learning_gamma_performance(iterations, lower_sample, upper_sample, H, nombre_archivo_historial, nombre_archivo_gammas, V, K, per_por, T, M, coor, Mk, Q, d, d_EV, q, q_EV, p, p_EV, c, max_cij, dist_demand_parm, r_EV, bo_EV, Km, dist_p, dist_q, best_gamma_ST, best_gamma_MY, d_log, p_log, q_log):
    
    upper = upper_sample #40
    lower = lower_sample #20
    
    flag_improvement = True

    #parejas = [(3,1), (3,0), (3,best_gamma_ST), (3, best_tiempo), (0, best_gamma_MY), (0, 0), (0, 1)]
    
    parejas = [(H,1), (H,0), (H,best_gamma_ST)]
    #parejas = [(H,best_gamma_ST)]
    
    method = 1
    historial = {}
    historial_gammas = {}
    dif_setting = {}
    
    for pareja in parejas:  
        H = pareja[0]
        gamma = [pareja[1] for i in M]
        
        print(H, pareja[1])        
        
        count_global = np.zeros(len(M))
        
        inc_FO_policy_obj = 0
        obj_history = []
        time_history = []
        gamma_history = []
        
        if flag_improvement == False:
              iterations = 1
              
        historial[(H, pareja[1])] = []
        historial_gammas[(H, pareja[1])] = []
        
        ev_dif = []
        for iteration in range(iterations):
            
            print(iteration)
        
            gamma_iter = np.zeros((upper-lower, len(M)))
            count_iter = np.zeros((upper-lower, len(M)))
        
            R_ST = []
            time_count1  = []
            
            gamma_history.append(gamma.copy())
            sample_dif = []
            for sample in range(lower, upper):
                #print(sample)
                
                tiempoInicio1 = time.time()
                final_policy_RS, indicators_RS, count_i_RS, gamma_i_RS = Stochasctic_Rolling_Horizon(Vertex, V, Products, K, per_por, Periods, T, M, Mk, Q, d, dist_demand_parm, q, dist_q, p, dist_p, c, max_cij, sample, sample_paths, H, gamma, r_EV, bo_EV, alpha, MIP_gap, d_log, p_log, q_log)
                tiempoOpti1 = time.time()-tiempoInicio1
                
                time_count1.append(tiempoOpti1)
                R_ST.append(indicators_RS[0])
                sample_dif.append(indicators_RS[7])
                
                for i in M:
                    gamma_iter[sample-lower, i-1] = gamma_i_RS[i]
                    count_iter[sample-lower, i-1] = count_i_RS[i]
            
            ev_dif.append(np.average(sample_dif))
            
            if flag_improvement == True:
                
                if method == 0:
                    
                    gamma = np.average(gamma_iter, axis = 0)
                    
                elif method == 1:
                    
                    count_ayuda = np.sum(count_iter, axis=0)
                    
                    for i in M:
                        if count_ayuda[i-1]>0:
                            count_global[i-1]+=1
                            
                    gamma_ayuda = np.average(gamma_iter, axis = 0)
                    
                    for i in M:
                        if count_global[i-1]>0:
                            
                            part_a = (1 - (1/np.sqrt(count_global[i-1])))*gamma[i-1]
                            part_b = (1/np.sqrt(count_global[i-1]))*gamma_ayuda[i-1]
                            
                            gamma[i-1] = round(part_a+part_b,3)
                        
            obj_history.append(round(np.average(R_ST),2))
            time_history.append(round(np.average(time_count1),2))
    
            if round(np.average(R_ST),2) > inc_FO_policy_obj:
                inc_gamma = gamma.copy()
                inc_FO_policy_obj = round(np.average(R_ST),2) 
                
        #print(gamma)
        #print(inc_gamma)
        
        dif_setting[H, pareja[1]] = ev_dif.copy()
        
        historial[(H, pareja[1])].append(obj_history)
        historial_gammas[(H, pareja[1])].append(gamma_history)
    
    with open(nombre_archivo_historial+'.pickle', 'wb') as handle:
        pickle.dump(historial, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    with open('Route_difference_tunning'+nombre_archivo_historial+'.pickle', 'wb') as handle:
        pickle.dump(dif_setting, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    #GAMMAS CON MYOPIC POLICY
    
    H = 0
    gamma = [best_gamma_MY for i in M]
    gamma_iter = np.zeros((upper-lower, len(M)))

    R_ST = []
    time_count1  = []

    for sample in range(lower, upper):
        #print(sample)
        
        tiempoInicio1 = time.time()
        final_policy_RS, indicators_RS, count_i_RS, gamma_i_RS = Stochasctic_Rolling_Horizon(Vertex, V, Products, K, per_por, Periods, T, M, Mk, Q, d, dist_demand_parm, q, dist_q, p, dist_p, c, max_cij, sample, sample_paths, H, gamma, r_EV, bo_EV, alpha, MIP_gap, d_log, p_log, q_log)
        tiempoOpti1 = time.time()-tiempoInicio1
        
        for i in M:
            gamma_iter[sample-lower, i-1] = gamma_i_RS[i]

    gamma_DDMY = np.average(gamma_iter, axis = 0)

    historial_gammas["DDMY"] = gamma_DDMY
    with open(nombre_archivo_gammas+'.pickle', 'wb') as handle:
        pickle.dump(historial_gammas, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
def theta_tunning_myopic(lower_sample, upper_sample, nombre_archivo, V, K, per_por, T, M, coor, Mk, Q, d, d_EV, q, q_EV, p, p_EV, c, max_cij, dist_demand_parm, r_EV, bo_EV, Km, dist_p, dist_q, best_gamma):
    upper = upper_sample #100   
    lower = lower_sample
    
    gamma = [best_gamma for i in M]
    thetas = [0.1, 0.25, 0.5, 1, 1.5, 2]
    stocha_obj_MY = []
    
    for index in range(len(thetas)):
        #print(thetas[index])

        R_MY = []

        for sample in range(lower, upper):
            final_policy_MY, indicators_MY = Deterministic_Rolling_Horizon(Vertex, V, Products, K, per_por, Periods, T, M, Mk, Q, d, d_EV,q, q_EV,p, p_EV, c, max_cij, sample, 1, gamma, r_EV, bo_EV, alpha, Km, thetas[index], MIP_gap, True)
            R_MY.append(indicators_MY[0])
        
        stocha_obj_MY.append(np.average(R_MY))
        
    final = [thetas, stocha_obj_MY]
        
    with open(nombre_archivo+'.pickle', 'wb') as handle:
        pickle.dump(final, handle, protocol=pickle.HIGHEST_PROTOCOL)
        


def Run_experiment_complexity_model(name_exp):
     
    nombre_aux3 =  'historial_gammas_'+name_exp
    nombre_aux4 = 'Final_Indicartors_Xij'+name_exp
    
    V, K, per_por, T, M, coor, Mk, Q, d, d_EV, q, q_EV, p, p_EV, c, max_cij, dist_demand_parm, r_EV, bo_EV, Km, dist_p, dist_q, avg_EV_p, avg_EV_q, d_log, p_log, q_log  = Genera_Instacias_Stochastic_with_parameters_fixed(Vertex, Products, Periods, paths_evaluation, veh_factor, cap_factor, seedd, increase_route, increase_offering, perish, name_exp)  
    
    with open('historial_gammas_General_NormxNorm.pickle', 'rb') as handle:
        Load_historial_gammas = pickle.load(handle)
        
    gammas_one  = Load_historial_gammas[(3, 1)][0][-1]
    
    lo = lo_eval
    up = up_eval
    
    H = H_aux
    V = range(V_aux)
    M = range(1, V_aux)
    
    for k in K:
        sup = []
        for i in Mk[k,0]:
            if i in M:
                sup.append(i)
        Mk[k,0] = sup
        
    
    for t in range(1, len(T)):
        for k in K:
            Mk[k,t] = Mk[k,0].copy()
    
    
    STxij_SC, STxij_SC_I = Build_indicators_policy('ST_Xij_'+name_exp, lo, up, gammas_one, H, 3, 0.5, V, K, per_por, T, M, coor, Mk, Q, d, d_EV, q, q_EV, p, p_EV, c, max_cij, dist_demand_parm, r_EV, bo_EV, Km, dist_p, dist_q, 0, False, d_log, p_log, q_log)
    
    ST_SC_1, ST_SC_1_I = Build_indicators_policy('ST_T1_'+name_exp, lo, up, gammas_one, H, 2, 0.5, V, K, per_por, T, M, coor, Mk, Q, d, d_EV, q, q_EV, p, p_EV, c, max_cij, dist_demand_parm, r_EV, bo_EV, Km, dist_p, dist_q, 0, False, d_log, p_log, q_log)
    
    #resultado_SC = [MY_SC_I, MY_SC_I_theta, DT_SC_I, ST_SC_I, ST_SC_1_I, ST_SC_0_I, ST_SC_BG_I, ST_SC_D_I, ST_SC_DDMY_I, ST_SC_CAP_I, ST_SC_PRICE_I, 0, 0]
    resultado_SC = [0, 0, 0, 0, ST_SC_1_I, STxij_SC_I, 0, 0, 0, 0, 0, 0, 0]
    
    with open(nombre_aux4+'.pickle', 'wb') as handle:
        pickle.dump(resultado_SC, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    #linea = [name_exp+"\t"+str(MY_SC)+"\t"+str(MY_SC_theta)+"\t"+str(DT_SC)+"\t"+str(ST_SC)+"\t"+str(ST_SC_1)+"\t"+str(ST_SC_0)+"\t"+str(ST_SC_BG)+"\t"+str(ST_SC_D)+"\t"+str(ST_SC_DDMY)+"\t"+str(ST_SC_CAP)+"\t"+str(ST_SC_PRICE)+"\t"+str(ST_SC_AVG3)+"\t"+str(ST_SC_AVG0BD)+"\n"]
    linea1 = [name_exp+"\t"+str(ST_SC_1[0])+"\t"+str(ST_SC_1[1])+"\t"+str(ST_SC_1[2])+"\t"+str(0)+"\t"+str(STxij_SC[0])+"\t"+str(STxij_SC[1])+"\t"+str(STxij_SC[2])+"\t"+str(0)+"\t"+str(0)+"\t"+str(0)+"\t"+str(0)+"\t"+str(0)+"\t"+str(0)+"\n"]
    #linea2 = ['time'+name_exp+"\t"+str(0)+"\t"+str(0)+"\t"+str(0)+"\t"+str(0)+"\t"+str(ST_SC_1[2])+"\t"+str(STxij_SC[1],STxij_SC[2])+"\t"+str(0)+"\t"+str(0)+"\t"+str(0)+"\t"+str(0)+"\t"+str(0)+"\t"+str(0)+"\t"+str(0)+"\n"]
    
    with open('ResumenExperimentos.txt', "a") as h:
        h.writelines(linea1)
        #h.writelines(linea2)
    


def Run_experiment(name_exp, bandera_cosas):
     
    nombre_aux = 'Obj_gammas_'+name_exp
    nombre_aux1 = 'Obj_times_'+name_exp
    nombre_aux2 =  'historial_'+name_exp
    nombre_aux3 =  'historial_gammas_'+name_exp
    
    nombre_aux2_ =  'historial_myopic_'+name_exp
    nombre_aux3_ =  'historial_gammas_myopic_'+name_exp
    
    nombre_aux4 = 'Final_Indicartors_'+name_exp
    nombre_aux5 = 'Thetas_myopic_'+name_exp
    nombre_aux6 = 'C_Daganzo_'+name_exp
    
    V, K, per_por, T, M, coor, Mk, Q, d, d_EV, q, q_EV, p, p_EV, c, max_cij, dist_demand_parm, r_EV, bo_EV, Km, dist_p, dist_q, avg_EV_p, avg_EV_q, d_log, p_log, q_log  = Genera_Instacias_Stochastic_with_parameters_fixed(Vertex, Products, Periods, paths_evaluation, veh_factor, cap_factor, seedd, increase_route, increase_offering, perish, name_exp)  
    
    split_1 = re.split('_', name_exp)
    split_2 = re.split('x', split_1[1])
    
    '''
    with open('Datos_instancia_general.pickle', 'rb') as handle:
        ins = pickle.load(handle)
    
    V, K, per_por, T, M, coor, Mk, Q, d, d_EV, q, q_EV, p, p_EV, c, max_cij, dist_demand_parm, r_EV, bo_EV, Km, dist_p, dist_q, avg_EV_p, avg_EV_q = ins[0], ins[1], ins[2], ins[3], ins[4], ins[5], ins[6], ins[7], ins[8], ins[9], ins[10], ins[11], ins[12], ins[13], ins[14], ins[15], ins[16], ins[17], ins[18], ins[19], ins[20], ins[21], ins[22], ins[23]
    '''
    
    #datos = [V, K, per_por, T, M, coor, Mk, Q, d, d_EV, q, q_EV, p, p_EV, c, max_cij, dist_demand_parm, r_EV, bo_EV, Km, dist_p, dist_q, avg_EV_p, avg_EV_q ]
    
    #with open('Datos_instancia_general.pickle', 'wb') as handle:
    #    pickle.dump(datos, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    Gammas_price_capacity(name_exp, M, Km, dist_q, avg_EV_q, dist_p, avg_EV_p)
    
    if perish == 2:
        K_ayuda = [k for k in K]
        np.random.shuffle(K_ayuda)
        
        per_por[K_ayuda[0]] = 0
        per_por[K_ayuda[1]] = 0
        per_por[K_ayuda[2]] = 0.2
        per_por[K_ayuda[3]] = 0.2
        per_por[K_ayuda[4]] = 0.1
    
    
    if bandera_cosas == True:
        tunning_C_daganzo(130, 150, nombre_aux6, V, K, per_por, T, M, coor, Mk, Q, d, d_EV, q, q_EV, p, p_EV, c, max_cij, dist_demand_parm, r_EV, bo_EV, Km, dist_p, dist_q, 0.5, d_log, p_log, q_log)
    
    split_3 = re.split('x', nombre_aux6)
    with open(split_3[0]+'x'+split_2[0]+'.pickle', 'rb') as handle:
        Load_C_daganzo = pickle.load(handle)
        
    key = max(Load_C_daganzo, key=Load_C_daganzo.get)
        
    C_daganzo = [key[0], key[1]] 
    print(C_daganzo)
    #Load_C_daganzo[0][np.argmax(Load_C_daganzo[1])]
    #print('C',C_daganzo)
    
    if bandera_cosas == True:
        H_size_and_gamma_performance(0, 20, 3, 4, nombre_aux, V, K, per_por, T, M, coor, Mk, Q, d, d_EV, q, q_EV, p, p_EV, c, max_cij, dist_demand_parm, r_EV, bo_EV, Km, dist_p, dist_q, d_log, p_log, q_log)
    
    split_3 = re.split('x', nombre_aux)
    with open(split_3[0]+'x'+split_2[0]+'.pickle', 'rb') as handle:
        Load_obj_gammas= pickle.load(handle)
        
    
    values_gammas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    
    MY = Load_obj_gammas[0]
    DT = Load_obj_gammas[1]
    ST = Load_obj_gammas[2]
    
    best_gamma_MY = values_gammas[np.argmax([MY[i][0] for i in range(len(values_gammas))])]
    best_gamma_DT = values_gammas[np.argmax([DT[i][0][0] for i in range(len(values_gammas))])]
    best_gamma_ST = values_gammas[np.argmax([ST[i][0][0] for i in range(len(values_gammas))])]
    
    if bandera_cosas == True:
        theta_tunning_myopic(120, 140, nombre_aux5, V, K, per_por, T, M, coor, Mk, Q, d, d_EV, q, q_EV, p, p_EV, c, max_cij, dist_demand_parm, r_EV, bo_EV, Km, dist_p, dist_q, best_gamma_MY)
    
    split_3 = re.split('x', nombre_aux5)
    with open(split_3[0]+'x'+split_2[0]+'.pickle', 'rb') as handle:
        Load_thetas = pickle.load(handle)
    
    B_theta = Load_thetas[0][np.argmax(Load_thetas[1])]
    print(B_theta)
    
    if bandera_cosas == True:
        gamma_distance_performance(30, 50, 3, nombre_aux1, V, K, per_por, T, M, coor, Mk, Q, d, d_EV, q, q_EV, p, p_EV, c, max_cij, dist_demand_parm, r_EV, bo_EV, Km, dist_p, dist_q, d_log, p_log, q_log)
    
    split_3 = re.split('x', nombre_aux1)
    with open(split_3[0]+'x'+split_2[0]+'.pickle', 'rb') as handle:
        Load_Obj_times_dist = pickle.load(handle)
    
    best_tiempo = Load_Obj_times_dist[0][0][np.argmax(Load_Obj_times_dist[0][1])]
    
    gammas_dist = []
    for i in M:
        cont = 0
        for j in M:
            if i != j:
                if c[i,j] <= best_tiempo:
                    cont+=1
        gammas_dist.append(1/(1+cont))
    
    if bandera_cosas == True: 
        #learing with stochastic lookahead
        Learning_gamma_performance(iterations_learning, 60, 80, 3, nombre_aux2, nombre_aux3, V, K, per_por, T, M, coor, Mk, Q, d, d_EV, q, q_EV, p, p_EV, c, max_cij, dist_demand_parm, r_EV, bo_EV, Km, dist_p, dist_q, best_gamma_ST, best_gamma_MY, d_log, p_log, q_log)
        
        #learning with myopic
        Learning_gamma_performance(iterations_learning, 60, 80, 0, nombre_aux2_, nombre_aux3_, V, K, per_por, T, M, coor, Mk, Q, d, d_EV, q, q_EV, p, p_EV, c, max_cij, dist_demand_parm, r_EV, bo_EV, Km, dist_p, dist_q, best_gamma_MY, best_gamma_MY, d_log, p_log, q_log)
    
    split_3 = re.split('x', nombre_aux3)
    with open(split_3[0]+'x'+split_2[0]+'.pickle', 'rb') as handle:
        Load_historial_gammas = pickle.load(handle)
        
    split_3 = re.split('x', nombre_aux3_)
    with open(split_3[0]+'x'+split_2[0]+'.pickle', 'rb') as handle:
        Load_historial_gammas_MY = pickle.load(handle)
        
    with open('Gammas_capacity_'+split_1[0]+'_'+split_2[0]+'x'+split_2[0]+'.pickle', 'rb') as handle:
        gammas_cap = pickle.load(handle)
        
    with open('Gammas_price_'+split_1[0]+'_'+split_2[0]+'x'+split_2[0]+'.pickle', 'rb') as handle:
        gammas_price = pickle.load(handle)
    
    gammas_one  = Load_historial_gammas[(3, 1)][0][-1]
    gammas_cero = Load_historial_gammas[(3, 0)][0][-1]
    gammas_BG = Load_historial_gammas[(3, best_gamma_ST)][0][-1]
    gamma_DDMY = Load_historial_gammas["DDMY"]
    
    
    gammas_one_MY  = Load_historial_gammas_MY[(0, 1)][0][-1]
    gammas_cero_MY = Load_historial_gammas_MY[(0, 0)][0][-1]
    gammas_BG_MY = Load_historial_gammas_MY[(0, best_gamma_MY)][0][-1]
    
    gammas_avg3 = [np.average([gammas_cap[i-1], gammas_price[i-1], gammas_dist[i-1]])  for i in M]
    #gammas_avg_0_bt = [(gammas_cero[i-1] + gammas_dist[i-1])/2  for i in M]

    lo = lo_eval
    up = up_eval
    
    H = 0
    MY_SC, MY_SC_I = Build_indicators_policy('MY_BG_'+name_exp, lo, up, [best_gamma_MY], H, 0, 0.5, V, K, per_por, T, M, coor, Mk, Q, d, d_EV, q, q_EV, p, p_EV, c, max_cij, dist_demand_parm, r_EV, bo_EV, Km, dist_p, dist_q, 0, False, d_log, p_log, q_log)
    MY_SC_1, MY_SC_1_I = Build_indicators_policy('MY_BG_T1'+name_exp, lo, up, gammas_one_MY, H, 0, 0.5, V, K, per_por, T, M, coor, Mk, Q, d, d_EV, q, q_EV, p, p_EV, c, max_cij, dist_demand_parm, r_EV, bo_EV, Km, dist_p, dist_q, 0, False, d_log, p_log, q_log)
    MY_SC_0, MY_SC_0_I = Build_indicators_policy('MY_BG_T0'+name_exp, lo, up, gammas_cero_MY, H, 0, 0.5, V, K, per_por, T, M, coor, Mk, Q, d, d_EV, q, q_EV, p, p_EV, c, max_cij, dist_demand_parm, r_EV, bo_EV, Km, dist_p, dist_q, 0, False, d_log, p_log, q_log)
    MY_SC_BG, MY_SC_BG_I = Build_indicators_policy('MY_BG_TBG'+name_exp, lo, up, gammas_BG_MY, H, 0, 0.5, V, K, per_por, T, M, coor, Mk, Q, d, d_EV, q, q_EV, p, p_EV, c, max_cij, dist_demand_parm, r_EV, bo_EV, Km, dist_p, dist_q, 0, False, d_log, p_log, q_log)
    
    H = 1
    MY_SC_theta, MY_SC_I_theta = Build_indicators_policy('MY_BG_Theta'+name_exp, lo, up, [best_gamma_MY], H, 0, 0.5, V, K, per_por, T, M, coor, Mk, Q, d, d_EV, q, q_EV, p, p_EV, c, max_cij, dist_demand_parm, r_EV, bo_EV, Km, dist_p, dist_q, B_theta, True, d_log, p_log, q_log)
    
    H = 3
    DT_SC, DT_SC_I = Build_indicators_policy('DT_BG_'+name_exp, lo, up, [best_gamma_DT], H, 1, 0.5, V, K, per_por, T, M, coor, Mk, Q, d, d_EV, q, q_EV, p, p_EV, c, max_cij, dist_demand_parm, r_EV, bo_EV, Km, dist_p, dist_q, 0, False, d_log, p_log, q_log)
    
    
    np.random.seed(seedd)
    instances = np.random.choice(100000000, replicates)
    #print(instances)
    
    resultado_SC = {}
    
    for rep in instances:
    
        ST_SC, ST_SC_I = Build_indicators_policy('ST_BG_'+name_exp, lo, up, [best_gamma_ST], H, 2, 0.5, V, K, per_por, T, M, coor, Mk, Q, d, d_EV, q, q_EV, p, p_EV, c, max_cij, dist_demand_parm, r_EV, bo_EV, Km, dist_p, dist_q, 0, False, d_log, p_log, q_log, rep)
        ST_SC_1, ST_SC_1_I = Build_indicators_policy('ST_T1_'+name_exp, lo, up, gammas_one, H, 2, 0.5, V, K, per_por, T, M, coor, Mk, Q, d, d_EV, q, q_EV, p, p_EV, c, max_cij, dist_demand_parm, r_EV, bo_EV, Km, dist_p, dist_q, 0, False, d_log, p_log, q_log, rep)
        ST_SC_0, ST_SC_0_I = Build_indicators_policy('ST_T0_'+name_exp, lo, up, gammas_cero, H, 2, 0.5, V, K, per_por, T, M, coor, Mk, Q, d, d_EV, q, q_EV, p, p_EV, c, max_cij, dist_demand_parm, r_EV, bo_EV, Km, dist_p, dist_q, 0, False, d_log, p_log, q_log, rep)
        ST_SC_BG, ST_SC_BG_I = Build_indicators_policy('ST_TBG_'+name_exp, lo, up, gammas_BG, H, 2, 0.5, V, K, per_por, T, M, coor, Mk, Q, d, d_EV, q, q_EV, p, p_EV, c, max_cij, dist_demand_parm, r_EV, bo_EV, Km, dist_p, dist_q, 0, False, d_log, p_log, q_log, rep)
        ST_SC_D, ST_SC_D_I = Build_indicators_policy('ST_D_'+name_exp, lo, up, gammas_dist, H, 2, 0.5, V, K, per_por, T, M, coor, Mk, Q, d, d_EV, q, q_EV, p, p_EV, c, max_cij, dist_demand_parm, r_EV, bo_EV, Km, dist_p, dist_q, 0, False, d_log, p_log, q_log, rep)
        ST_SC_DDMY, ST_SC_DDMY_I = Build_indicators_policy('ST_DDMY_'+name_exp, lo, up, gamma_DDMY, H, 2, 0.5, V, K, per_por, T, M, coor, Mk, Q, d, d_EV, q, q_EV, p, p_EV, c, max_cij, dist_demand_parm, r_EV, bo_EV, Km, dist_p, dist_q, 0, False, d_log, p_log, q_log, rep)
        ST_SC_CAP, ST_SC_CAP_I = Build_indicators_policy('ST_CAP_'+name_exp, lo, up, gammas_cap, H, 2, 0.5, V, K, per_por, T, M, coor, Mk, Q, d, d_EV, q, q_EV, p, p_EV, c, max_cij, dist_demand_parm, r_EV, bo_EV, Km, dist_p, dist_q, 0, False, d_log, p_log, q_log, rep)
        ST_SC_PRICE, ST_SC_PRICE_I = Build_indicators_policy('ST_PRICE_'+name_exp, lo, up, gammas_price, H, 2, 0.5, V, K, per_por, T, M, coor, Mk, Q, d, d_EV, q, q_EV, p, p_EV, c, max_cij, dist_demand_parm, r_EV, bo_EV, Km, dist_p, dist_q, 0, False, d_log, p_log, q_log, rep)
        ST_SC_AVG3, ST_SC_AVG3_I = Build_indicators_policy('ST_AVG3_'+name_exp, lo, up, gammas_avg3, H, 2, 0.5, V, K, per_por, T, M, coor, Mk, Q, d, d_EV, q, q_EV, p, p_EV, c, max_cij, dist_demand_parm, r_EV, bo_EV, Km, dist_p, dist_q, 0, False, d_log, p_log, q_log, rep)
        ST_SC_DAG, ST_SC_DAG_I = Build_indicators_policy('ST_DAG_'+name_exp, lo, up, [best_gamma_ST], H, 4, 0.5, V, K, per_por, T, M, coor, Mk, Q, d, d_EV, q, q_EV, p, p_EV, c, max_cij, dist_demand_parm, r_EV, bo_EV, Km, dist_p, dist_q, C_daganzo, False, d_log, p_log, q_log, rep)
        
        #resultado_SC[rep] = [MY_SC_I, MY_SC_I_theta, DT_SC_I, ST_SC_I, ST_SC_1_I, ST_SC_0_I, ST_SC_BG_I, ST_SC_D_I, ST_SC_DDMY_I, ST_SC_CAP_I, ST_SC_PRICE_I, ST_SC_AVG3_I, MY_SC_1_I, MY_SC_0_I, MY_SC_BG_I, 0] #ST_SC_DAG_I
        resultado_SC[rep] = [MY_SC_I, MY_SC_I_theta, DT_SC_I, ST_SC_I, ST_SC_1_I, ST_SC_0_I, ST_SC_BG_I, ST_SC_D_I, ST_SC_DDMY_I, ST_SC_CAP_I, ST_SC_PRICE_I, ST_SC_AVG3_I, MY_SC_1_I, MY_SC_0_I, MY_SC_BG_I, ST_SC_DAG_I]
        
        #linea = [name_exp+"\t"+str(MY_SC)+"\t"+str(MY_SC_theta)+"\t"+str(DT_SC)+"\t"+str(ST_SC)+"\t"+str(ST_SC_1)+"\t"+str(ST_SC_0)+"\t"+str(ST_SC_BG)+"\t"+str(ST_SC_D)+"\t"+str(ST_SC_DDMY)+"\t"+str(ST_SC_CAP)+"\t"+str(ST_SC_PRICE)+"\t"+str(ST_SC_AVG3)+"\t"+str(ST_SC_AVG0BD)+"\n"]
        
        #linea1 = [name_exp+str(rep)+"\t"+str(MY_SC[0])+"\t"+str(MY_SC_theta[0])+"\t"+str(DT_SC[0])+"\t"+str(ST_SC[0])+"\t"+str(ST_SC_1[0])+"\t"+str(ST_SC_0[0])+"\t"+str(ST_SC_BG[0])+"\t"+str(ST_SC_D[0])+"\t"+str(ST_SC_DDMY[0])+"\t"+str(ST_SC_CAP[0])+"\t"+str(ST_SC_PRICE[0])+"\t"+str(ST_SC_AVG3[0])+"\t"+str(MY_SC_1[0])+"\t"+str(MY_SC_0[0])+"\t"+str(MY_SC_BG[0])+"\t"+str(0)+"\n"] #+str(ST_SC_DAG[0])+"\n"]
        #linea2 = ['time'+name_exp+str(rep)+"\t"+str(MY_SC[1])+"\t"+str(MY_SC_theta[1])+"\t"+str(DT_SC[1])+"\t"+str(ST_SC[1])+"\t"+str(ST_SC_1[1])+"\t"+str(ST_SC_0[1])+"\t"+str(ST_SC_BG[1])+"\t"+str(ST_SC_D[1])+"\t"+str(ST_SC_DDMY[1])+"\t"+str(ST_SC_CAP[1])+"\t"+str(ST_SC_PRICE[1])+"\t"+str(ST_SC_AVG3[1])+"\t"+str(MY_SC_1[1])+"\t"+str(MY_SC_0[1])+"\t"+str(MY_SC_BG[1])+"\t"+str(0)+"\n"] #+str(ST_SC_DAG[1])+"\n"]
        
        linea1 = [name_exp+str(rep)+"\t"+str(MY_SC[0])+"\t"+str(MY_SC_theta[0])+"\t"+str(DT_SC[0])+"\t"+str(ST_SC[0])+"\t"+str(ST_SC_1[0])+"\t"+str(ST_SC_0[0])+"\t"+str(ST_SC_BG[0])+"\t"+str(ST_SC_D[0])+"\t"+str(ST_SC_DDMY[0])+"\t"+str(ST_SC_CAP[0])+"\t"+str(ST_SC_PRICE[0])+"\t"+str(ST_SC_AVG3[0])+"\t"+str(MY_SC_1[0])+"\t"+str(MY_SC_0[0])+"\t"+str(MY_SC_BG[0])+"\t"+str(ST_SC_DAG[0])+"\n"]
        linea2 = ['time'+name_exp+str(rep)+"\t"+str(MY_SC[1])+"\t"+str(MY_SC_theta[1])+"\t"+str(DT_SC[1])+"\t"+str(ST_SC[1])+"\t"+str(ST_SC_1[1])+"\t"+str(ST_SC_0[1])+"\t"+str(ST_SC_BG[1])+"\t"+str(ST_SC_D[1])+"\t"+str(ST_SC_DDMY[1])+"\t"+str(ST_SC_CAP[1])+"\t"+str(ST_SC_PRICE[1])+"\t"+str(ST_SC_AVG3[1])+"\t"+str(MY_SC_1[1])+"\t"+str(MY_SC_0[1])+"\t"+str(MY_SC_BG[1])+"\t"+str(ST_SC_DAG[1])+"\n"]
        
        
        with open('ResumenExperimentos.txt', "a") as h:
            h.writelines(linea1)
            h.writelines(linea2)
    
    with open(nombre_aux4+'.pickle', 'wb') as handle:
        pickle.dump(resultado_SC, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
        
def Gammas_price_capacity(nombre, M, Km, dist_q, avg_EV_q, dist_p, avg_EV_p):
    gamma_cap = []
    gamma_price = []
    for i in M:
        count_cap = 1
        count_pri = 1
        for k in Km[i]:
            if dist_q[i,k][0] > avg_EV_q[k]:
                count_cap+=1
                
            if dist_p[i,k][0] < avg_EV_p[k]:
                count_pri+=1
        
        valor1 = 1/count_cap
        valor2 = 1/count_pri
        
        gamma_cap.append(valor1)
        gamma_price.append(valor2)
    
    with open('Gammas_capacity_'+nombre+'.pickle', 'wb') as handle:
        pickle.dump(gamma_cap, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open('Gammas_price_'+nombre+'.pickle', 'wb') as handle:
        pickle.dump(gamma_price, handle, protocol=pickle.HIGHEST_PROTOCOL)


def distribution_experiments():
    crea_datos = True
    calibra = True

    dist_train = 'Norm'
    dist_eval = 'Norm'
    name_exp = 'General_'+dist_train+'x'+dist_eval
    Run_experiment(name_exp, calibra)

    dist_train = 'LogNorm'
    dist_eval = 'LogNorm'
    name_exp = 'General_'+dist_train+'x'+dist_eval
    Run_experiment(name_exp, calibra)

    dist_train = 'Triang'
    dist_eval = 'Triang'
    name_exp = 'General_'+dist_train+'x'+dist_eval
    Run_experiment(name_exp, calibra)

    dist_train = 'Uniform'
    dist_eval = 'Uniform'
    name_exp = 'General_'+dist_train+'x'+dist_eval
    Run_experiment(name_exp, calibra)

    #_________________________________________________________

    crea_datos = False
    calibra = False

    dist_train = 'Norm'
    dist_eval = 'LogNorm'
    name_exp = 'General_'+dist_train+'x'+dist_eval
    Run_experiment(name_exp, calibra)

    dist_train = 'Norm'
    dist_eval = 'Triang'
    name_exp = 'General_'+dist_train+'x'+dist_eval
    Run_experiment(name_exp, calibra)

    dist_train = 'Norm'
    dist_eval = 'Uniform'
    name_exp = 'General_'+dist_train+'x'+dist_eval
    Run_experiment(name_exp, calibra)

    #_________________________________________________________

    dist_train = 'LogNorm'
    dist_eval = 'Norm'
    name_exp = 'General_'+dist_train+'x'+dist_eval
    Run_experiment(name_exp, calibra)

    dist_train = 'LogNorm'
    dist_eval = 'Triang'
    name_exp = 'General_'+dist_train+'x'+dist_eval
    Run_experiment(name_exp, calibra)

    dist_train = 'LogNorm'
    dist_eval = 'Uniform'
    name_exp = 'General_'+dist_train+'x'+dist_eval
    Run_experiment(name_exp, calibra)

    #_________________________________________________________

    dist_train = 'Triang'
    dist_eval = 'Norm'
    name_exp = 'General_'+dist_train+'x'+dist_eval
    Run_experiment(name_exp, calibra)

    dist_train = 'Triang'
    dist_eval = 'LogNorm'
    name_exp = 'General_'+dist_train+'x'+dist_eval
    Run_experiment(name_exp, calibra)

    dist_train = 'Triang'
    dist_eval = 'Uniform'
    name_exp = 'General_'+dist_train+'x'+dist_eval
    Run_experiment(name_exp, calibra)

    #_________________________________________________________

    dist_train = 'Uniform'
    dist_eval = 'Norm'
    name_exp = 'General_'+dist_train+'x'+dist_eval
    Run_experiment(name_exp, calibra)

    dist_train = 'Uniform'
    dist_eval = 'LogNorm'
    name_exp = 'General_'+dist_train+'x'+dist_eval
    Run_experiment(name_exp, calibra)

    dist_train = 'Uniform'
    dist_eval = 'Triang'
    name_exp = 'General_'+dist_train+'x'+dist_eval
    Run_experiment(name_exp, calibra)

def experiments_longer_route():
    crea_datos = True
    calibra = True

    dist_train = 'Norm'
    dist_eval = 'Norm'

    complex_route_values = [0.5, 1, 1.5, 2, 4]

    copia_daily_time = 480
    copia_veh_factor = 60


    for value in complex_route_values:
        name_exp = 'RoutesSize'+str(value)+'_'+dist_train+'x'+dist_eval
        daily_time = copia_daily_time*value
        veh_factor = copia_veh_factor*value
        
        Run_experiment(name_exp, calibra)
        
def experiment_complex_model():
    flag_demand_cero_t_0 = False #True: Demand first period is cero
    T_aux = True #solo voy a considerar el primer periodo del horizonte
    dist_train = 'Norm'
    dist_eval = 'Norm'

    for V_aux in [3, 6, 11, 21]:
        for H_aux in [0, 1, 2, 3]:        
            name_exp = 'RouteComplexity_'+str(V_aux-1)+"_"+str(H_aux)
            Run_experiment_complexity_model(name_exp)
            
            
def correlation_per_period_experiments():
    levels = 2 # si es 2 solo se mira nivel alto, o bajo, si es 3, se mira alto, normaly bajo

    #bandera para correlacionar el precio y la oferta por periodo
    prices_correlated_time = True
    offer_correlated_time = False

    name_exp = 'General'+str(prices_correlated_time)+str(offer_correlated_time)+str(levels)+'_'+dist_train+'x'+dist_eval
    Run_experiment(name_exp, calibra)

    prices_correlated_time = False
    offer_correlated_time = True

    name_exp = 'General'+str(prices_correlated_time)+str(offer_correlated_time)+str(levels)+'_'+dist_train+'x'+dist_eval
    Run_experiment(name_exp, calibra)

    prices_correlated_time = True
    offer_correlated_time = True

    name_exp = 'General'+str(prices_correlated_time)+str(offer_correlated_time)+str(levels)+'_'+dist_train+'x'+dist_eval
    Run_experiment(name_exp, calibra)

    levels = 3 # si es 2 solo se mira nivel alto, o bajo, si es 3, se mira alto, normaly bajo

    #bandera para correlacionar el precio y la oferta por periodo
    prices_correlated_time = True
    offer_correlated_time = False

    name_exp = 'General'+str(prices_correlated_time)+str(offer_correlated_time)+str(levels)+'_'+dist_train+'x'+dist_eval
    Run_experiment(name_exp, calibra)

    prices_correlated_time = False
    offer_correlated_time = True

    name_exp = 'General'+str(prices_correlated_time)+str(offer_correlated_time)+str(levels)+'_'+dist_train+'x'+dist_eval
    Run_experiment(name_exp, calibra)

    prices_correlated_time = True
    offer_correlated_time = True

    name_exp = 'General'+str(prices_correlated_time)+str(offer_correlated_time)+str(levels)+'_'+dist_train+'x'+dist_eval
    Run_experiment(name_exp, calibra)
    
    
def route_heuristic_comparison():
    flag_routing = 'EXACT' #main algorithm to routing
    
    heuristic_dif =  'SPLIT' #select the routing algorithm to compare
    flag_dif_routing = True #compare the solution of this heuristic with the main heuristic
    
    name_exp = 'General'+flag_routing+heuristic_dif+str(flag_dif_routing)+'_'+dist_train+'x'+dist_eval
    Run_experiment(name_exp, calibra)
    
    
def base_experiments_parameters():
    #-------------------------------------------------- PRICE DISTRIBUTION ---------------------------------------

    std_price_values = [0, 0.2]

    for std_price in std_price_values:  

        name_exp = 'stdPrice'+str(std_price)+'_'+dist_train+'x'+dist_eval
        Run_experiment(name_exp, calibra)
        
    std_price = 0.1    
        
    #------------------------------------------------- AVAILABLE QUANTITY ---------------------------------------

    std_quiantity_values = [0, 0.2]

    for std_quiantity in std_quiantity_values:  

        name_exp = 'stdQuiantity'+str(std_quiantity)+'_'+dist_train+'x'+dist_eval
        Run_experiment(name_exp, calibra)
        
    std_quiantity = 0.1

    #------------------------------------------------ DEMANDA DISTRIBUTION ---------------------------------------

    std_demand_values = [0, 0.2]

    for std_demand in std_demand_values:  

        name_exp = 'stdDemand'+str(std_demand)+'_'+dist_train+'x'+dist_eval
        Run_experiment(name_exp, calibra)

    std_demand  = 0.1

    #-------------------------------------------------- CORRELATED PRICES ---------------------------------------

    flag_correlated_price = True
    std_price_values = [0, 0.2, 0.1]

    for std_price in std_price_values:  
        
        name_exp = 'correlatedPrices'+str(flag_correlated_price)+str(std_price)+'_'+dist_train+'x'+dist_eval
        Run_experiment(name_exp, calibra)
        
    flag_correlated_price = False
    std_price = 0.1

    #-------------------------------------------------- CORRELATED AVAILABLE QUANTITY ---------------------------------------

    flag_correlated_cap = True
    std_quiantity_values = [0, 0.2, 0.1]

    for std_quiantity in std_quiantity_values:  

        name_exp = 'correlatedQuiantity'+str(flag_correlated_cap)+str(std_quiantity)+'_'+dist_train+'x'+dist_eval
        Run_experiment(name_exp, calibra)

    flag_correlated_cap = False
    std_quiantity = 0.1   
    

#Grafica_Ruta(coor, V)
#H_size_and_gamma_performance(70, 100, 1, 9, 'CalibraciónFull') Calibracion completa de ventanas 
#Gap_No_samplepaths_performance(100, 120, "Gap_SamplePaths_Calibration") Calibracion completa de gap's y sample paths

Vertex = 21
Products = 5
Periods = 20

paths_evaluation = 200
sample_paths = 10
size_grid = 250
MIP_gap = 0.05

seedd = 10

alpha = 0.5
daily_time = 480 #
time_per_supplier = 10
flag_demand_cero_t_0 = True #True: Demand first period is cero

veh_factor = 60 #vehicle capacity in hundreds of kilograms
increase_offering = 0.3 
cap_factor = 1
increase_route = 1 #a minute is worth a dollar
revenue_dist = 0.1 #Incremento en revenue

#Price distribution information
lower_price = 50 #$ dollars per hundred kg
upper_price = 120 
std_price = 0.1 # 10 #12% of average

#Available quantity distribution information
lower_quiantity = 4 #hundreds of kilograms
upper_quiantity  = 9
std_quiantity = 0.1 #2 #24% of average

#Demand distribution information
lower_demand = 10  #hundreds of kilograms
upper_demand= 15 
std_demand = 0.1 #3 #24% of average

lkh_time_limit = 2
solver_name = 'gurobi' #solver for set-partitioning

flag_correlated_price = False
flag_correlated_cap = False
T_aux = False #hace que solo se ejecute el primer estado del problema

perish = 0.1
flag_routing = 'SPLIT' #main algorithm to routing
iterations_learning = 60 

lo_eval = 90
up_eval = 120

heuristic_dif =  'EXACT' #select the routing algorithm to compare
flag_dif_routing = False #compare the solution of this heuristic with the main heuristic

dist_train = 'Norm'
dist_eval = 'Norm'

prices_correlated_time = False
offer_correlated_time = False
levels = None

replicates = 1

calibra = False
crea_datos = False

dist_train = 'Norm'
dist_eval = 'Norm'

complex_route_values = [1]

copia_daily_time = 480
copia_veh_factor = 60


for value in complex_route_values:
    name_exp = 'RoutesSize'+str(value)+'_'+dist_train+'x'+dist_eval
    daily_time = copia_daily_time*value
    veh_factor = copia_veh_factor*value
    
    Run_experiment(name_exp, calibra)








 
