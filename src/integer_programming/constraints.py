import gurobipy as gp

def ul_in_yard(m): 
    # for every unit load and in every time step the sum of all b and g vars is at most 1
    for n in m.Unit_loads:
        for t in range(1, m.T):
            constr_name = f"ul_in_yard_n{n.get_id()}_t{t}"
            m.model.addConstr(
                gp.quicksum(m.b_vars[(i.get_ap_id(), j.get_id(), n.get_id(), t)] for i in m.Lanes[:-1] for j in i.get_tiers()) + 
                m.g_vars[(n.get_id(), t)] <= 1,
                name=constr_name)

def ul_in_yard_dm(m): 
    # for every unit load and in every time step the sum of all b vars is at most the value of s var
    for n in m.Unit_loads:
        for t in range(1, m.T):
            constr_name = f"ul_in_yard_n{n.get_id()}_t{t}"
            m.model.addConstr(
                gp.quicksum(m.b_vars[(i.get_ap_id(), j.get_id(), n.get_id(), t)] for i in m.Lanes[1:-1] for j in i.get_tiers()) <= m.s_vars[(n.get_id(), t)],
                name=constr_name)

def ul_in_yard2_dm(m):
    # for every unit load and in every time step the g var is greater equal than the s var - sum of b vars
    for n in m.Unit_loads:
        for t in range(1, m.T):
            constr_name = f"ul_in_yard2_n{n.get_id()}_t{t}"
            m.model.addConstr(
                m.g_vars[(n.get_id(), t)] >= m.s_vars[(n.get_id(), t)] - gp.quicksum(m.b_vars[(i.get_ap_id(), j.get_id(), n.get_id(), t)] for i in m.Lanes[1:-1] for j in i.get_tiers()),
                name=constr_name)

def at_most_ul_in_slot(m):
    # for every slot and in every time step the sum of all b vars is at most 1
    for i in m.Lanes[:-1]:
        for j in i.get_tiers():
            for t in range(1, m.T):
                constr_name = f"max_1_ul_lane{i.get_ap_id()}_tier{j.get_id()}_time{t}"
                m.model.addConstr(
                    gp.quicksum(m.b_vars[(i.get_ap_id(), j.get_id(), n.get_id(), t)] for n in m.Unit_loads) <= 1,
                    name=constr_name)

def at_most_ul_in_slot_dm(m):
    # for every slot and in every time step the sum of all b vars is at most 1
    for i in m.Lanes[1:-1]:
        for j in i.get_tiers():
            for t in range(1, m.T):
                constr_name = f"max_1_ul_lane{i.get_ap_id()}_tier{j.get_id()}_time{t}"
                m.model.addConstr(
                    gp.quicksum(m.b_vars[(i.get_ap_id(), j.get_id(), n.get_id(), t)] for n in m.Unit_loads) <= 1,
                    name=constr_name)

def no_hollow_spaces(m):
    # prevents hollow spaces in the lanes by ensuring that a good can only be placed into a tier if the tier below is occupied
    for i in m.Lanes[:-1]:
        for j in i.get_tiers()[:-1]:
            for t in range(1, m.T):
                constr_name = f"no_hollow_spaces_lane{i.get_ap_id()}_tier{j.get_id()}_time{t}"
                m.model.addConstr(gp.quicksum(m.b_vars[(i.get_ap_id(), j.get_id(), n.get_id(), t)] for n in m.Unit_loads) >= 
                                  gp.quicksum(m.b_vars[(i.get_ap_id(), j.get_id()+1, n.get_id(), t)] for n in m.Unit_loads),
                                  name=constr_name)

def no_hollow_spaces_dm(m):
    # prevents hollow spaces in the lanes by ensuring that a good can only be placed into a tier if the tier below is occupied
    for i in m.Lanes[1:-1]:
        for j in i.get_tiers()[:-1]:
            for t in range(1, m.T):
                constr_name = f"no_hollow_spaces_lane{i.get_ap_id()}_tier{j.get_id()}_time{t}"
                m.model.addConstr(gp.quicksum(m.b_vars[(i.get_ap_id(), j.get_id(), n.get_id(), t)] for n in m.Unit_loads) >= 
                                  gp.quicksum(m.b_vars[(i.get_ap_id(), j.get_id()+1, n.get_id(), t)] for n in m.Unit_loads),
                                  name=constr_name)

def one_move_per_vehicle(m):
    # limits each time step to a single retrieval or relocation for each available vehicle at that position
    for i in m.Lanes[:-1]:
        for j in i.get_tiers():
            for t in range(1, m.T):
                constr_name = f"one_move_per_vehicle_lane{i.get_ap_id()}_tier{j.get_id()}_time{t}"
                m.model.addConstr(
                    gp.quicksum(m.x_vars[(i.get_ap_id(), j.get_id(), k.get_ap_id(), l.get_id(), n.get_id(), t)] for k in m.Lanes[:-1] for l in k.get_tiers() for n in m.Unit_loads) + 
                    gp.quicksum(m.e_vars[(i.get_ap_id(), j.get_id(), k.get_ap_id(), l.get_id(), t)] for k in m.Lanes for l in k.get_tiers()) + 
                    gp.quicksum(m.y_vars[(i.get_ap_id(), j.get_id(), n.get_id(), t)] for n in m.Unit_loads) == m.c_vars[(i.get_ap_id(), j.get_id(), t)], 
                    name=constr_name)

def one_move_per_vehicle_dm(m):
    # limits each time step to a single retrieval or relocation for each available vehicle at that position
    for i in m.Lanes[1:-1]:
        for j in i.get_tiers():
            for t in range(1, m.T):
                for v in m.Vehicles:
                    constr_name = f"one_move_per_vehicle_lane{i.get_ap_id()}_tier{j.get_id()}_time{t}_vehicle{v.get_id()}"
                    m.model.addConstr(
                        gp.quicksum(m.x_vars[(i.get_ap_id(), j.get_id(), k.get_ap_id(), l.get_id(), n.get_id(), t, v.get_id())] for k in m.Lanes[1:-1] for l in k.get_tiers() for n in m.Unit_loads) + 
                        gp.quicksum(m.e_vars[(i.get_ap_id(), j.get_id(), k.get_ap_id(), l.get_id(), t, v.get_id())] for k in m.Lanes for l in k.get_tiers()) + 
                        gp.quicksum(m.y_vars[(i.get_ap_id(), j.get_id(), n.get_id(), t, v.get_id())] for n in m.Unit_loads)  
                        <= m.c_vars[(i.get_ap_id(), j.get_id(), t, v.get_id())], 
                        name=constr_name)

def one_move_per_vehicle_sink(m): 
    # limits each time step to a single repositioning for each vehicle at the sink
    I = m.Lanes[-1] 
    j = I.get_tiers()[0]
    for t in range(1, m.T):
        constr_name = f"one_move_per_vehicle_sink_time{t}"
        m.model.addConstr(
            gp.quicksum(m.e_vars[(I.get_ap_id(), j.get_id(), k.get_ap_id(), l.get_id(), t)] for k in m.Lanes for l in k.get_tiers()) == m.c_vars[(I.get_ap_id(), j.get_id(), t)], 
            name=constr_name)

def one_move_per_vehicle_sink_dm(m): 
    # limits each time step to a single repositioning for each vehicle at the sink
    I = m.Lanes[-1] 
    j = I.get_tiers()[0]
    for t in range(1, m.T):
        for v in m.Vehicles:
            constr_name = f"one_move_per_vehicle_sink_time{t}_vehicle{v.get_id()}"
            m.model.addConstr(
                gp.quicksum(m.e_vars[(I.get_ap_id(), j.get_id(), k.get_ap_id(), l.get_id(), t, v.get_id())] for k in m.Lanes for l in k.get_tiers()) 
                <= m.c_vars[(I.get_ap_id(), j.get_id(), t, v.get_id())], 
                name=constr_name)

def one_move_per_vehicle_source_dm(m): 
    # limits each time step to a single storage move for each vehicle at the source
    source = m.Lanes[0]
    s_tier = source.get_tiers()[0]
    for t in range(1, m.T): 
        for v in m.Vehicles: 
            constr_name = f"one_move_per_vehicle_source_time{t}_vehicle{v.get_id()}"
            m.model.addConstr(
                gp.quicksum(m.z_vars[(k.get_ap_id(), l.get_id(), n.get_id(), t, v.get_id())] for k in m.Lanes[1:-1] for l in k.get_tiers() for n in m.Unit_loads) +
                gp.quicksum(m.y_vars[(source.get_ap_id(), s_tier.get_id(), n.get_id(), t, v.get_id())] for n in m.Unit_loads) +
                gp.quicksum(m.e_vars[(source.get_ap_id(), s_tier.get_id(), k.get_ap_id(), l.get_id(), t, v.get_id())] for k in m.Lanes for l in k.get_tiers()) <=
                m.c_vars[(source.get_ap_id(), s_tier.get_id(), t, v.get_id())],
                name=constr_name)

def direct_retrieval_if_not_stored_dm(m): 
    # ensures that a unit load can only be directly retrieved if it is not stored
    source = m.Lanes[0]
    for t in range(1, m.T):
        for n in m.Unit_loads:
                constr_name = f"direct_retrieval_if_not_stored_ul{n.get_id()}_time{t}"
                m.model.addConstr(
                    gp.quicksum(m.y_vars[(source.get_ap_id(), source.get_tiers()[0].get_id(), n.get_id(), t, v.get_id())] for v in m.Vehicles) +
                m.s_vars[n.get_id(), t] <= 1, name=constr_name)

def one_move_per_ul(m):
    # limits each time step to move or retrieve as many unit loads as there are available
    for i in m.Lanes[:-1]: 
        for j in i.get_tiers():
            for n in m.Unit_loads:
                for t in range(1, m.T):
                    constr_name = f"one_move_per_ul_lane{i.get_ap_id()}_tier{j.get_id()}_ul{n.get_id()}_time{t}"
                    m.model.addConstr(
                        m.y_vars[(i.get_ap_id(), j.get_id(), n.get_id(), t)] + 
                        gp.quicksum(m.x_vars[(i.get_ap_id(), j.get_id(), k.get_ap_id(), l.get_id(), n.get_id(), t)] for k in m.Lanes[:-1] for l in k.get_tiers()) <= 
                        m.b_vars[(i.get_ap_id(), j.get_id(), n.get_id(), t)], 
                        name=constr_name)

def one_move_per_ul_dm(m):
    # limits each time step to move or retrieve as many unit loads as there are available
    for i in m.Lanes[1:-1]: 
        for j in i.get_tiers():
            for n in m.Unit_loads:
                for t in range(1, m.T):
                    constr_name = f"one_move_per_ul_lane{i.get_ap_id()}_tier{j.get_id()}_ul{n.get_id()}_time{t}"
                    m.model.addConstr(
                        gp.quicksum(m.y_vars[(i.get_ap_id(), j.get_id(), n.get_id(), t, v.get_id())] for v in m.Vehicles) + 
                        gp.quicksum(m.x_vars[(i.get_ap_id(), j.get_id(), k.get_ap_id(), l.get_id(), n.get_id(), t, v.get_id())] 
                                    for k in m.Lanes[1:-1] 
                                    for l in k.get_tiers()
                                    for v in m.Vehicles)
                                    <= m.b_vars[(i.get_ap_id(), j.get_id(), n.get_id(), t)], 
                        name=constr_name)

def config_update(m):
    # ensures that the configuration of the buffer is updated after every move
    for i in m.Lanes[:-1]:
        for j in i.get_tiers():
            for t in range(2, m.T):  
                for n in m.Unit_loads:
                    constr_name = f"config_update_lane{i.get_ap_id()}_tier{j.get_id()}_time{t}_ul{n.get_id()}"
                    # Calculate the sum of relocations into the current slot
                    relocations_into_slot = gp.quicksum(
                        m.x_vars[k.get_ap_id(), l.get_id(), i.get_ap_id(), j.get_id(), n.get_id(), t_prime] 
                        for k in m.Lanes[:-1] for l in k.get_tiers() for t_prime in [t - m.calculate_travel_time(k, l, i, j, True)]
                        if t_prime >= 1  # Include only if the resulting index is valid (i.e., not less than 1)
                    )

                    # Calculate the sum of relocations out of the current slot
                    relocations_out_of_slot = gp.quicksum(
                        m.x_vars[i.get_ap_id(), j.get_id(), k.get_ap_id(), l.get_id(), n.get_id(), t - 1]
                        for k in m.Lanes[:-1] for l in k.get_tiers()
                    )

                    # Update configuration with valid indices only
                    m.model.addConstr(
                        m.b_vars[i.get_ap_id(), j.get_id(), n.get_id(), t] ==
                        m.b_vars[i.get_ap_id(), j.get_id(), n.get_id(), t - 1] +
                        relocations_into_slot -
                        relocations_out_of_slot -
                        m.y_vars[i.get_ap_id(), j.get_id(), n.get_id(), t - 1], 
                        name=constr_name
                    )

def config_update_dm(m):
    # ensures that the configuration of the buffer is updated after every move
    for i in m.Lanes[1:-1]:
        for j in i.get_tiers():
            for t in range(2, m.T):  
                for n in m.Unit_loads:
                    constr_name = f"config_update_lane{i.get_ap_id()}_tier{j.get_id()}_time{t}_ul{n.get_id()}"
                    # Calculate the sum of relocations into the current slot
                    relocations_into_slot = gp.quicksum(
                        m.x_vars[k.get_ap_id(), l.get_id(), i.get_ap_id(), j.get_id(), n.get_id(), t_prime, v.get_id()] 
                        for k in m.Lanes[1:-1] 
                        for l in k.get_tiers() 
                        for v in m.Vehicles
                        for t_prime in [t - m.calculate_travel_time(k, l, i, j, True)]
                        if t_prime >= 1  # Include only if the resulting index is valid (i.e., not less than 1)
                    )
                    # Calculate the sum of relocations out of the current slot
                    relocations_out_of_slot = gp.quicksum(
                        m.x_vars[i.get_ap_id(), j.get_id(), k.get_ap_id(), l.get_id(), n.get_id(), t - 1, v.get_id()]
                        for k in m.Lanes[1:-1] 
                        for l in k.get_tiers()
                        for v in m.Vehicles
                    )
                    # Calculate the sum of retrievals
                    retrievals = gp.quicksum(
                        m.y_vars[i.get_ap_id(), j.get_id(), n.get_id(), t - 1, v.get_id()]
                        for v in m.Vehicles
                    )
                    # Calculate the sum of storages
                    storages = gp.quicksum(
                        m.z_vars[i.get_ap_id(), j.get_id(), n.get_id(), t_prime, v.get_id()]
                        for v in m.Vehicles
                        for t_prime in [t - m.calculate_travel_time(i, j, "source", None, True)]
                        if t_prime >= 1  # Include only if the resulting index is valid (i.e., not less than 1)
                    )

                    # Update configuration with valid indices only
                    m.model.addConstr(
                        m.b_vars[i.get_ap_id(), j.get_id(), n.get_id(), t] ==
                        m.b_vars[i.get_ap_id(), j.get_id(), n.get_id(), t - 1] +
                        relocations_into_slot -
                        relocations_out_of_slot -
                        retrievals + 
                        storages, 
                        name=constr_name
                    )
                    # print(f"{m.b_vars[i.get_ap_id(), j.get_id(), n.get_id(), t]} = {storages}")

def vehicle_update(m):
    # ensures that the vehicle is updated after every move
    for i in m.Lanes[:-1]:
        for j in i.get_tiers():
            for t in range(2, m.T):  
                constr_name = f"vehicle_update_lane{i.get_ap_id()}_tier{j.get_id()}_time{t}"
                # Calculate the sum of retrievals from the current vehicle slot
                retrievals_from_slot_vehicle = gp.quicksum(m.y_vars[i.get_ap_id(), j.get_id(), n.get_id(), t - 1] for n in m.Unit_loads)
                # Calculate the sum of relocations into the current vehicle slot
                relocations_into_slot_vehicle = gp.quicksum(
                    m.x_vars[k.get_ap_id(), l.get_id(), i.get_ap_id(), j.get_id(), n.get_id(), t_prime] 
                    for k in m.Lanes[:-1] for l in k.get_tiers() for n in m.Unit_loads for t_prime in [t - m.calculate_travel_time(k, l, i, j, True)]
                    if t_prime >= 1  # Include only if the resulting index is valid
                )
                # Calculate the sum of repositionings out of the current vehicle slot
                repositionings_into_slot_vehicle = gp.quicksum(
                    m.e_vars[k.get_ap_id(), l.get_id(), i.get_ap_id(), j.get_id(), t_prime]
                    for k in m.Lanes for l in k.get_tiers() for t_prime in [t - m.calculate_travel_time(k, l, i, j, False)]
                    if t_prime >= 1  # Include only if the resulting index is valid
                )
                # Calculate the sum of relocations out of the current vehicle slot
                relocations_out_of_slot_vehicle = gp.quicksum(
                    m.x_vars[i.get_ap_id(), j.get_id(), k.get_ap_id(), l.get_id(), n.get_id(), t - 1]
                    for k in m.Lanes[:-1] for l in k.get_tiers() for n in m.Unit_loads
                )
                # Calculate the sum of repositionings out of the current vehicle slot
                repositionings_out_of_slot_vehicle = gp.quicksum(
                    m.e_vars[i.get_ap_id(), j.get_id(), k.get_ap_id(), l.get_id(), t - 1]
                    for k in m.Lanes for l in k.get_tiers()
                )
                # Update the vehicle configuration with valid indices only
                m.model.addConstr(
                    m.c_vars[i.get_ap_id(), j.get_id(), t] ==
                    m.c_vars[i.get_ap_id(), j.get_id(), t - 1] -
                    retrievals_from_slot_vehicle +
                    relocations_into_slot_vehicle +
                    repositionings_into_slot_vehicle -
                    relocations_out_of_slot_vehicle -
                    repositionings_out_of_slot_vehicle,
                    name=constr_name
                )

def vehicle_update_dm(m):
    # ensures that the vehicle is updated after every move
    for i in m.Lanes[1:-1]:
        for j in i.get_tiers():
            for t in range(2, m.T):  
                for v in m.Vehicles:
                    constr_name = f"vehicle_update_lane{i.get_ap_id()}_tier{j.get_id()}_time{t}_vehicle{v.get_id()}"
                    # Calculate the sum of storages into the current vehicle slot
                    storages_into_slot_vehicle = gp.quicksum(
                        m.z_vars[i.get_ap_id(), j.get_id(), n.get_id(), t_prime, v.get_id()] for n in m.Unit_loads
                        for t_prime in [t - m.calculate_travel_time(i, j, "source", None, True)] if t_prime >= 1  # Include only if the resulting index is valid
                    )
                    # Calculate the sum of retrievals from the current vehicle slot
                    retrievals_from_slot_vehicle = gp.quicksum(m.y_vars[i.get_ap_id(), j.get_id(), n.get_id(), t - 1, v.get_id()] for n in m.Unit_loads)
                    # Calculate the sum of relocations into the current vehicle slot
                    relocations_into_slot_vehicle = gp.quicksum(
                        m.x_vars[k.get_ap_id(), l.get_id(), i.get_ap_id(), j.get_id(), n.get_id(), t_prime, v.get_id()] 
                        for k in m.Lanes[1:-1] for l in k.get_tiers() for n in m.Unit_loads for t_prime in [t - m.calculate_travel_time(k, l, i, j, True)]
                        if t_prime >= 1  # Include only if the resulting index is valid
                    )
                    # Calculate the sum of repositionings into the current vehicle slot
                    repositionings_into_slot_vehicle = gp.quicksum(
                        m.e_vars[k.get_ap_id(), l.get_id(), i.get_ap_id(), j.get_id(), t_prime, v.get_id()]
                        for k in m.Lanes for l in k.get_tiers() for t_prime in [t - m.calculate_travel_time(k, l, i, j, False)]
                        if t_prime >= 1  # Include only if the resulting index is valid
                    )
                    # Calculate the sum of relocations out of the current vehicle slot
                    relocations_out_of_slot_vehicle = gp.quicksum(
                        m.x_vars[i.get_ap_id(), j.get_id(), k.get_ap_id(), l.get_id(), n.get_id(), t - 1, v.get_id()]
                        for k in m.Lanes[1:-1] for l in k.get_tiers() for n in m.Unit_loads
                    )
                    # Calculate the sum of repositionings out of the current vehicle slot
                    repositionings_out_of_slot_vehicle = gp.quicksum(
                        m.e_vars[i.get_ap_id(), j.get_id(), k.get_ap_id(), l.get_id(), t - 1, v.get_id()]
                        for k in m.Lanes for l in k.get_tiers()
                    )
                    # Update the vehicle configuration with valid indices only
                    m.model.addConstr(
                        m.c_vars[i.get_ap_id(), j.get_id(), t, v.get_id()] ==
                        m.c_vars[i.get_ap_id(), j.get_id(), t - 1, v.get_id()] + 
                        storages_into_slot_vehicle -
                        retrievals_from_slot_vehicle +
                        relocations_into_slot_vehicle +
                        repositionings_into_slot_vehicle -
                        relocations_out_of_slot_vehicle -
                        repositionings_out_of_slot_vehicle,
                        name=constr_name
                    )
                    # if t==33 and i.get_ap_id() == 2 and j.get_id() == 1:
                    #     print(f"{m.c_vars[i.get_ap_id(), j.get_id(), t, v.get_id()]} = {m.c_vars[i.get_ap_id(), j.get_id(), t - 1, v.get_id()]} + ({storages_into_slot_vehicle}) - ({retrievals_from_slot_vehicle}) + ({relocations_into_slot_vehicle}) + ({repositionings_into_slot_vehicle}) - ({relocations_out_of_slot_vehicle}) - ({repositionings_out_of_slot_vehicle})")

def vehicle_update_sink(m):
    # ensures that the vehicle is updated after every move to the sink
    I = m.Lanes[-1]  # Assuming that I is the sink
    for t in range(2, m.T): 
        constr_name = f"vehicle_update_sink_time{t}"
        # Calculate the sum of moves into the sink, ensuring that the time index is valid
        retrievals = gp.quicksum(
            m.y_vars[i.get_ap_id(), j.get_id(), n.get_id(), t_prime] 
            for n in m.Unit_loads 
            for i in m.Lanes[:-1]
            for j in i.get_tiers()
            for t_prime in [t - m.calculate_travel_time(i, j, I, 1, True)] 
            if t_prime >= 1
        )

        # Calculate the sum of repositionings into the sink, ensuring that the time index is valid
        repositions_into_sink = gp.quicksum(
            m.e_vars[i.get_ap_id(), j.get_id(), I.get_ap_id(), 1, t_prime]
            for i in m.Lanes
            for j in i.get_tiers()
            for t_prime in [t - m.calculate_travel_time(i, j, I, 1, False)] if t_prime >= 1
        )
        
        # Calculate the sum of repositionings from the sink, ensuring that the time index is valid
        repositionings_from_sink = gp.quicksum(
            m.e_vars[I.get_ap_id(), 1, i.get_ap_id(), j.get_id(), t - 1]
            for i in m.Lanes
            for j in i.get_tiers()
        )
        # Update the vehicle configuration at the sink with valid indices only
        m.model.addConstr(
            m.c_vars[I.get_ap_id(), 1, t] == 
            m.c_vars[I.get_ap_id(), 1, t - 1] +
            retrievals +
            repositions_into_sink -
            repositionings_from_sink,
            name = constr_name
        )

def vehicle_update_sink_dm(m):
    # ensures that the vehicle is updated after every move to the sink
    I = m.Lanes[-1]  # Assuming that I is the sink
    for t in range(2, m.T): 
        for v in m.Vehicles:
            constr_name = f"vehicle_update_sink_time{t}_vehicle{v.get_id()}"
            # Calculate the sum of moves into the sink, ensuring that the time index is valid
            retrievals = gp.quicksum(
                m.y_vars[i.get_ap_id(), j.get_id(), n.get_id(), t_prime, v.get_id()] 
                for n in m.Unit_loads 
                for i in m.Lanes[:-1]
                for j in i.get_tiers()
                for t_prime in [t - m.calculate_travel_time(i, j, I, 1, True)] 
                if t_prime >= 1
            )

            # Calculate the sum of repositionings into the sink, ensuring that the time index is valid
            repositions_into_sink = gp.quicksum(
                m.e_vars[i.get_ap_id(), j.get_id(), I.get_ap_id(), 1, t_prime, v.get_id()]
                for i in m.Lanes
                for j in i.get_tiers()
                for t_prime in [t - m.calculate_travel_time(i, j, I, 1, False)] if t_prime >= 1
            )
            
            # Calculate the sum of repositionings from the sink, ensuring that the time index is valid
            repositionings_from_sink = gp.quicksum(
                m.e_vars[I.get_ap_id(), 1, i.get_ap_id(), j.get_id(), t - 1, v.get_id()]
                for i in m.Lanes
                for j in i.get_tiers()
            )
            # Update the vehicle configuration at the sink with valid indices only
            m.model.addConstr(
                m.c_vars[I.get_ap_id(), 1, t, v.get_id()] == 
                m.c_vars[I.get_ap_id(), 1, t - 1, v.get_id()] +
                retrievals +
                repositions_into_sink -
                repositionings_from_sink,
                name = constr_name
            )

def vehicle_update_source_dm(m):
    # ensures that the vehicle is updated after every move to the source
    source = m.Lanes[0]  
    source_tier = source.get_tiers()[0]
    for t in range(2, m.T):
        for v in m.Vehicles:
            constr_name = f"vehicle_update_source_time{t}_vehicle{v.get_id()}"
            # Calculate the sum of moves into the source, ensuring that the time index is valid
            storages = gp.quicksum(
                m.z_vars[i.get_ap_id(), j.get_id(), n.get_id(), t - 1, v.get_id()] 
                for n in m.Unit_loads 
                for i in m.Lanes[1:-1]
                for j in i.get_tiers()
            )
            repositions_into_source = gp.quicksum(
                m.e_vars[i.get_ap_id(), j.get_id(), source.get_ap_id(), source_tier.get_id(), t_prime, v.get_id()]
                for i in m.Lanes
                for j in i.get_tiers()
                for t_prime in [t - m.calculate_travel_time(i, j, source, None, False)] if t_prime >= 1
            )
            repositions_from_source = gp.quicksum(
                m.e_vars[source.get_ap_id(), source_tier.get_id(), i.get_ap_id(), j.get_id(), t - 1, v.get_id()]
                for i in m.Lanes
                for j in i.get_tiers()
            )
            direct_retrievals = gp.quicksum(
                m.y_vars[source.get_ap_id(), source_tier.get_id(), n.get_id(), t - 1, v.get_id()]
                for n in m.Unit_loads
            )
            # Update the vehicle configuration at the source with valid indices only
            m.model.addConstr(
                m.c_vars[source.get_ap_id(), source.get_tiers()[0].get_id(), t, v.get_id()] == 
                m.c_vars[source.get_ap_id(), source.get_tiers()[0].get_id(), t - 1, v.get_id()] -  
                storages + 
                repositions_into_source -
                repositions_from_source - 
                direct_retrievals,
                name = constr_name
            )


def relations_retrieval_config_vars(m):
    # establishes a connection between the configuration and the retrieval variables
    for n in m.Unit_loads:
        for t in range(1, m.T):
            constr_name = f"relations_retrieval_config_vars_ul{n.get_id()}_time{t}"
            m.model.addConstr(
                m.g_vars[n.get_id(), t] == gp.quicksum(m.y_vars[i.get_ap_id(), j.get_id(), n.get_id(), t_bar] 
                                                        for i in m.Lanes[:-1] 
                                                        for j in i.get_tiers()
                                                        for t_bar in range(1, t)),  # Include t-1 in the summation
                name=constr_name
            )

def relations_retrieval_config_vars_dm(m):
    # establishes a connection between the configuration and the retrieval variables
    for n in m.Unit_loads:
        for t in range(1, m.T):
            constr_name = f"relations_retrieval_config_vars_ul{n.get_id()}_time{t}"
            m.model.addConstr(
                m.g_vars[n.get_id(), t] == gp.quicksum(m.y_vars[i.get_ap_id(), j.get_id(), n.get_id(), t_bar, v.get_id()] 
                                                        for i in m.Lanes[:-1] 
                                                        for j in i.get_tiers()
                                                        for t_bar in range(1, t)  # Include t-1 in the summation
                                                        for v in m.Vehicles),
                name=constr_name
            )

def relations_storage_config_vars_dm(m):
    # establishes a connection between the configuration and the storage variables
    for n in m.Unit_loads:
        if n.stored is False: 
            for t in range(1, m.T):
                constr_name = f"relations_storage_config_vars_ul{n.get_id()}_time{t}"
                relations_storage_constraint = 0 
                for i in m.Lanes[1:-1]:
                    for j in i.get_tiers():
                        travel_time = m.calculate_travel_time(i, j, 'source', None, True)
                        for t_bar in range(1, t - travel_time + 1):
                            if t_bar >= 1:
                                for v in m.Vehicles:
                                    relations_storage_constraint += m.z_vars[i.get_ap_id(), j.get_id(), n.get_id(), t_bar, v.get_id()]
                m.model.addConstr(m.s_vars[n.get_id(), t] == m.s_vars[n.get_id(), 1] + relations_storage_constraint, name=constr_name)
                                

def retrieval_after_arrival(m):
    # Unit loads can only be retrieved after start of the retrieval window, considering travel time
    for n in m.Unit_loads:
        for i in m.Lanes[:-1]:
            for j in i.get_tiers():
                travel_time = m.calculate_travel_time(i, j, 'sink', 1, True)
                for t in range(1, n.get_retrieval_start() - travel_time):
                    if t >= 1:
                        m.model.addConstr(m.y_vars[i.get_ap_id(), j.get_id(), n.get_id(), t] == 0,
                                          name=f"retrieval_after_arrival_ul{n.get_id()}_{i.get_ap_id()}_{j.get_id()}_{t}")

def retrieval_after_arrival_dm(m):
    # Unit loads can only be retrieved after start of the retrieval window, considering travel time
    for n in m.Unit_loads:
        for v in m.Vehicles:
            for i in m.Lanes[:-1]:
                for j in i.get_tiers():
                    travel_time = m.calculate_travel_time(i, j, 'sink', 1, True)
                    for t in range(1, n.get_retrieval_start() - travel_time):
                        if t >= 1:
                            m.model.addConstr(m.y_vars[i.get_ap_id(), j.get_id(), n.get_id(), t, v.get_id()] == 0,
                                            name=f"retrieval_after_arrival_ul{n.get_id()}_{i.get_ap_id()}_{j.get_id()}_{t}_vehicle{v.get_id()}")

def retrieval_in_window(m):
    # Unit loads can only be retrieved within the retrieval window, considering travel time
    for n in m.Unit_loads:
        window_constraint = 0
        for i in m.Lanes[:-1]:
            for j in i.get_tiers():
                travel_time = m.calculate_travel_time(i, j, 'sink', 1, True)
                for t in range(n.get_retrieval_start() - travel_time, n.get_retrieval_end() - travel_time + 1):
                    if t >= 1:
                        window_constraint += m.y_vars[i.get_ap_id(), j.get_id(), n.get_id(), t]
        m.model.addConstr(window_constraint == 1, name=f"retrieval_in_window_ul{n.get_id()}")

def retrieval_in_window_dm(m):
    # Unit loads can only be retrieved within the retrieval window, considering travel time
    for n in m.Unit_loads:
        window_constraint = 0
        for i in m.Lanes[:-1]:
            for j in i.get_tiers():
                travel_time = m.calculate_travel_time(i, j, 'sink', 1, True)
                for t in range(n.get_retrieval_start() - travel_time, n.get_retrieval_end() - travel_time + 1):
                    if t >= 1:
                        for v in m.Vehicles:
                            window_constraint += m.y_vars[i.get_ap_id(), j.get_id(), n.get_id(), t, v.get_id()]
        m.model.addConstr(window_constraint == 1, name=f"retrieval_in_window_ul{n.get_id()}")

def retrieval_before_deadline(m):
    # Unit loads can only be retrieved before the deadline, considering travel time
    for n in m.Unit_loads:
        for i in m.Lanes[:-1]:
            for j in i.get_tiers():
                travel_time = m.calculate_travel_time(i, j, 'sink', 1, True)
                for t in range(n.get_retrieval_end() - travel_time + 1, m.T):
                    if t >= 1:
                        m.model.addConstr(m.y_vars[i.get_ap_id(), j.get_id(), n.get_id(), t] == 0,
                                          name=f"retrieval_before_deadline_ul{n.get_id()}_{i.get_ap_id()}_{j.get_id()}_{t}")

def retrieval_before_deadline_dm(m):
    # Unit loads can only be retrieved before the deadline, considering travel time
    for n in m.Unit_loads:
        for v in m.Vehicles:
            for i in m.Lanes[:-1]:
                for j in i.get_tiers():
                    travel_time = m.calculate_travel_time(i, j, 'sink', 1, True)
                    for t in range(n.get_retrieval_end() - travel_time + 1, m.T):
                        if t >= 1:
                            m.model.addConstr(m.y_vars[i.get_ap_id(), j.get_id(), n.get_id(), t, v.get_id()] == 0,
                                            name=f"retrieval_before_deadline_ul{n.get_id()}_{i.get_ap_id()}_{j.get_id()}_{t}_vehicle{v.get_id()}")

def stack_after_arrival_dm(m):
    # Unit loads can only be stacked after start of the stacking window, considering travel time
    for n in m.Unit_loads:
        if n.stored is False: 
            for v in m.Vehicles:
                for i in m.Lanes[1:-1]:
                    for j in i.get_tiers():
                        for t in range(1, n.get_arrival_start()):
                            if t >= 1:
                                m.model.addConstr(m.z_vars[i.get_ap_id(), j.get_id(), n.get_id(), t, v.get_id()] == 0,
                                                name=f"stack_after_arrival_ul{n.get_id()}_{i.get_ap_id()}_{j.get_id()}_{t}_vehicle{v.get_id()}")

def stack_in_window_dm(m):
    # Unit loads can only be stacked within the stacking window, considering travel time
    source = m.Lanes[0]
    travel_time_source_sink = m.calculate_travel_time(source, 1, 'sink', 1, True)
    for n in m.Unit_loads:
        if n.stored is False: 
            window_constraint = 0
            for i in m.Lanes[1:-1]:
                for j in i.get_tiers():
                    for t in range(n.get_arrival_start(), n.get_arrival_end() + 1):
                        if t >= 1:
                            for v in m.Vehicles:
                                window_constraint += m.z_vars[i.get_ap_id(), j.get_id(), n.get_id(), t, v.get_id()]
            # add direct retrieval from source
            for t in range(n.get_retrieval_start() - travel_time_source_sink, n.get_retrieval_end() - travel_time_source_sink + 1):
                if t >= 1 and t <= n.get_arrival_end():
                    for v in m.Vehicles:
                        window_constraint += m.y_vars[source.get_ap_id(), 1, n.get_id(), t, v.get_id()]
            m.model.addConstr(window_constraint == 1, name=f"stack_in_window_ul{n.get_id()}")

def stack_before_deadline_dm(m):
    # Unit loads can only be stacked before the deadline, considering travel time 
    for n in m.Unit_loads:
        if n.stored is False: 
            for v in m.Vehicles:
                for i in m.Lanes[1:-1]:
                    for j in i.get_tiers():
                        for t in range(n.get_arrival_end() + 1, m.T):
                            if t >= 1:
                                m.model.addConstr(m.z_vars[i.get_ap_id(), j.get_id(), n.get_id(), t, v.get_id()] == 0,
                                                name=f"stack_before_deadline_ul{n.get_id()}_{i.get_ap_id()}_{j.get_id()}_{t}_vehicle{v.get_id()}")

def one_vehicle_per_lane_dm(m): 
    # limits each time step to a single vehicle in each lane
    for i in m.Lanes[1:-1]:
        for t in range(1, m.T):
            constr_name = f"one_vehicle_per_lane_lane{i.get_ap_id()}_time{t}"
            m.model.addConstr( 
                gp.quicksum(m.c_vars[(i.get_ap_id(), j.get_id(), t, v.get_id())] for v in m.Vehicles for j in i.get_tiers()) <= 1, 
                    name=constr_name)

def lane_monopolization(m): 
    import math
    # Ensures that for every lane and every time step, the lane is occupied by at most one vehicle
    # Occupancy includes:
    # 1. Stationary vehicle (c_vars)
    # 2. Incoming vehicle (traveling in the lane)
    # 3. Outgoing vehicle (traveling in the lane)
    
    h = m.instance.get_handling_time()
    
    for i in m.Lanes[1:-1]: # Buffer lanes only
        J_i = len(i.get_tiers())
        
        # Pre-calculate travel times for each tier
        tier_travel = {}
        for j in i.get_tiers():
            dist = max(0, J_i - j.get_id()) 
            t_travel = math.ceil(dist / m.vehicle_speed)
            tier_travel[j.get_id()] = t_travel

        for t in range(1, m.T):
            constr_name = f"lane_monopolization_lane{i.get_ap_id()}_time{t}"
            
            # 1. Stationary vehicles
            stationary = gp.quicksum(m.c_vars[(i.get_ap_id(), j.get_id(), t, v.get_id())] 
                                     for j in i.get_tiers() for v in m.Vehicles)
            
            incoming_terms = []
            outgoing_terms = []

            for j in i.get_tiers():
                t_in = tier_travel[j.get_id()]
                t_out = tier_travel[j.get_id()]
                
                # --- Incoming Moves ---
                
                # z (Store): Source -> i,j. Duration: t_in + h
                duration = t_in + h
                tt_z = m.calculate_travel_time(m.Lanes[0], m.Lanes[0].get_tiers()[0], i, j, False)
                win_start = max(1, t - tt_z + 1)
                win_end = min(m.T - 1, t + duration - tt_z)
                
                for v in m.Vehicles:
                    for n in m.Unit_loads:
                        for t_s in range(win_start, win_end + 1):
                            if (i.get_ap_id(), j.get_id(), n.get_id(), t_s, v.get_id()) in m.z_vars:
                                incoming_terms.append(m.z_vars[(i.get_ap_id(), j.get_id(), n.get_id(), t_s, v.get_id())])

                # x (Relocate To i,j): From k,l -> i,j. Duration: t_in + h
                duration = t_in + h
                for k in m.Lanes[1:-1]:
                    for l in k.get_tiers():
                        tt_x = m.calculate_travel_time(k, l, i, j, False)
                        win_start = max(1, t - tt_x + 1)
                        win_end = min(m.T - 1, t + duration - tt_x)
                        for v in m.Vehicles:
                            for n in m.Unit_loads:
                                for t_s in range(win_start, win_end + 1):
                                    if (k.get_ap_id(), l.get_id(), i.get_ap_id(), j.get_id(), n.get_id(), t_s, v.get_id()) in m.x_vars:
                                        incoming_terms.append(m.x_vars[(k.get_ap_id(), l.get_id(), i.get_ap_id(), j.get_id(), n.get_id(), t_s, v.get_id())])

                # e (Reposition To i,j): From k,l -> i,j. Duration: t_in
                duration = t_in
                for k in m.Lanes: 
                    for l in k.get_tiers():
                        tt_e = m.calculate_travel_time(k, l, i, j, False)
                        win_start = max(1, t - tt_e + 1)
                        win_end = min(m.T - 1, t + duration - tt_e)
                        for v in m.Vehicles:
                            for t_s in range(win_start, win_end + 1):
                                if (k.get_ap_id(), l.get_id(), i.get_ap_id(), j.get_id(), t_s, v.get_id()) in m.e_vars:
                                    incoming_terms.append(m.e_vars[(k.get_ap_id(), l.get_id(), i.get_ap_id(), j.get_id(), t_s, v.get_id())])

                # --- Outgoing Moves ---
                
                # y (Retrieve): i,j -> Sink. Duration: h + t_out
                duration = h + t_out
                win_start = max(1, t - duration)
                win_end = min(m.T - 1, t - 1)
                
                for v in m.Vehicles:
                    for n in m.Unit_loads:
                        for t_s in range(win_start, win_end + 1):
                            if (i.get_ap_id(), j.get_id(), n.get_id(), t_s, v.get_id()) in m.y_vars:
                                outgoing_terms.append(m.y_vars[(i.get_ap_id(), j.get_id(), n.get_id(), t_s, v.get_id())])
                                    
                # x (Relocate From i,j): i,j -> k,l. Duration: h + t_out
                duration = h + t_out
                win_start = max(1, t - duration)
                win_end = min(m.T - 1, t - 1)
                for k in m.Lanes[1:-1]:
                    for l in k.get_tiers():
                        for v in m.Vehicles:
                            for n in m.Unit_loads:
                                for t_s in range(win_start, win_end + 1):
                                    if (i.get_ap_id(), j.get_id(), k.get_ap_id(), l.get_id(), n.get_id(), t_s, v.get_id()) in m.x_vars:
                                        outgoing_terms.append(m.x_vars[(i.get_ap_id(), j.get_id(), k.get_ap_id(), l.get_id(), n.get_id(), t_s, v.get_id())])

                # e (Reposition From i,j): i,j -> k,l. Duration: t_out
                duration = t_out
                win_start = max(1, t - duration)
                win_end = min(m.T - 1, t - 1)
                for k in m.Lanes:
                    for l in k.get_tiers():
                        for v in m.Vehicles:
                            for t_s in range(win_start, win_end + 1):
                                if (i.get_ap_id(), j.get_id(), k.get_ap_id(), l.get_id(), t_s, v.get_id()) in m.e_vars:
                                    outgoing_terms.append(m.e_vars[(i.get_ap_id(), j.get_id(), k.get_ap_id(), l.get_id(), t_s, v.get_id())])

            m.model.addConstr(stationary + gp.quicksum(incoming_terms) + gp.quicksum(outgoing_terms) <= 1, name=constr_name)

def lifo(m): 
    # ensures that the last unit load to be stored is the first to be retrieved
    for i in m.Lanes[1:-1]:  # Exclude source and sink lanes
        for j in i.get_tiers()[:-1]:  # Exclude the last tier as it has no tier behind it
            for t in range(1, m.T):
                for v in m.Vehicles:
                    m.model.addConstr(
                        gp.quicksum(m.x_vars[i.get_ap_id(), j.get_id(), k.get_ap_id(), l.get_id(), n.get_id(), t, v.get_id()] 
                                    for k in m.Lanes[1:-1] for l in k.get_tiers() for n in m.Unit_loads) + 
                        gp.quicksum(m.y_vars[i.get_ap_id(), j.get_id(), n.get_id(), t, v.get_id()] for n in m.Unit_loads) +
                        gp.quicksum(m.e_vars[i.get_ap_id(), j.get_id(), k.get_ap_id(), l.get_id(), t, v.get_id()] 
                                    for k in m.Lanes for l in k.get_tiers())
                        <= 1 - gp.quicksum(m.b_vars[i.get_ap_id(), j.get_id() + 1, n.get_id(), t] for n in m.Unit_loads),
                        name=f"lifo_lane{i.get_ap_id()}_tier{j.get_id()}_time{t}_vehicle{v.get_id()}"
                    )