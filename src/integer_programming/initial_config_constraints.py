import gurobipy as gp

def unit_load_start(m): 
    """
    ensures that the unit loads are at the start position at time 1
    get the layout of the warehouse from the initial bay state and
    set the initial position of the unit loads
    """
    t = 1
    for i in m.Lanes[:-1]:
        for j in i.get_tiers():
            for n in m.Unit_loads:
                constr_name = f"unit_load_start_lane{i.get_ap_id()}_tier{j.get_id()}_time{t}_ul{n.get_id()}"
                if j.get_initial_ul_id() == n.get_id():
                    m.model.addConstr(m.b_vars[i.get_ap_id(), j.get_id(), n.get_id(), t] == 1, name=constr_name)
                else: 
                    m.model.addConstr(m.b_vars[i.get_ap_id(), j.get_id(), n.get_id(), t] == 0, name=constr_name)

def unit_load_start_dm(m): 
    """
    ensures that the unit loads are at the start position at time 1
    get the layout of the warehouse from the initial bay state and
    set the initial position of the unit loads
    """
    t = 1
    for n in m.Unit_loads:
        constr_name2 = f"unit_load_stored_time{t}_ul{n.get_id()}"
        # print(f"{n}, {n.stored}")
        if n.stored is True: 
            m.model.addConstr(m.s_vars[n.get_id(), t] == 1, name=constr_name2)
        else: 
            m.model.addConstr(m.s_vars[n.get_id(), t] == 0, name=constr_name2)
        for i in m.Lanes[1:-1]:
            for j in i.get_tiers():
                constr_name = f"unit_load_start_lane{i.get_ap_id()}_tier{j.get_id()}_time{t}_ul{n.get_id()}"
                if j.get_initial_ul_id() == n.get_id():
                    m.model.addConstr(m.b_vars[i.get_ap_id(), j.get_id(), n.get_id(), t] == 1, name=constr_name)
                else: 
                    m.model.addConstr(m.b_vars[i.get_ap_id(), j.get_id(), n.get_id(), t] == 0, name=constr_name)

def vehicle_start(m):
    # ensures that the vehicle is at the start position (sink) at time 1
    t = 1
    I = m.Lanes[-1]
    for i in m.Lanes:
        for j in i.get_tiers():
            constr_name = f"vehicle_start_lane{i.get_ap_id()}_tier{j.get_id()}_time{t}"
            if i.get_ap_id() == I.get_ap_id() and j.get_id() == 1:
                m.model.addConstr(m.c_vars[i.get_ap_id(), j.get_id(), t] == 1, name=constr_name)
            else: 
                m.model.addConstr(m.c_vars[i.get_ap_id(), j.get_id(), t] == 0, name=constr_name)

def vehicle_start_dm(m):
    # ensures that the vehicle is at the start position (sink) at time 1
    t = 1
    I = m.Lanes[-1]
    for v in m.Vehicles: 
        for i in m.Lanes:
            for j in i.get_tiers():
                constr_name = f"vehicle_start_lane{i.get_ap_id()}_tier{j.get_id()}_time{t}"
                if i.get_ap_id() == I.get_ap_id() and j.get_id() == 1:
                    m.model.addConstr(m.c_vars[i.get_ap_id(), j.get_id(), t, v.get_id()] == 1, name=constr_name)
                else: 
                    m.model.addConstr(m.c_vars[i.get_ap_id(), j.get_id(), t, v.get_id()] == 0, name=constr_name)