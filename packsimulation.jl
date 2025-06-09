using Agents

# need to calibrate magnitudes of things (just in case vel too big, go out of bounds)
size = (10.0, 10.0)
space = ContinuousSpace(size; periodic = false;)

@agent struct Wolf(ContinuousAgent{2, Float64})
    group::Int
end

@agent struct Sheep(ContinuousAgent{2, Float64})
    group::Int
end

model = StandardABM(Wolf, space; properties, agent_step!, rng,
container = Vector, # agents are not removed, so we use this
scheduler = Schedulers.Randomly() # all agents are activated once at random
)

# will put in constructor; for now, putting hard-coded stuff here
min_safe_distance = 0.1
ww_force_coefficient = 0.5
sw_force_coefficient = 2 # force of sheep on wolf
dt = 1

function wolf_step!(predator, prey, model)

    current_distance = norm(predator.pos - prey.pos)
    
    if (current_distance <= min_safe_distance)

        acceleration = [0, 0]

        # for each neighbor wolf, find repulsive force α distance
        for neighbor in nearby_agents(prey, model)
            
            # sum up the repulsive force vectors to get acceleration
            distance_between = norm(predator.pos - neighbor.pos)
            acceleration += ww_force_coefficient * (predator.pos - neighbor.pos) / (distance_between)^2
        
        end

        # update velocity with v = v₀ + at
        predator.vel += acceleration * dt

    else
        # velocity here only depending on sheep attraction
        predator.vel = ((predator.pos - prey.pos)/current_distance)/dt
    
    end

    move_agent!(predator, model, dt)

    return
end