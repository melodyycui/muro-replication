using Agents
using LinearAlgebra

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

        wolf_repulsion = [0, 0]
        rotation_matrix = [cos(pi/2) sin(pi/2); -sin(pi/2) cos(pi/2)]

        # for each neighbor wolf, find repulsive force α distance
        for neighbor in nearby_agents(prey, model)
            
            # sum up the repulsive force vectors to get acceleration
            distance_between = norm(predator.pos - neighbor.pos)
            wolf_repulsion += ww_force_coefficient * (predator.pos - neighbor.pos) / (distance_between)^2

        end

        # projecting the repulsive force vector onto the tangential vector
        # to determine the direction the wolf travels along the circle
        u = rotation_matrix * (predator.pos - prey.pos)
        dot_product = dot(u, wolf_repulsion)
        proj_u_v = (dot_product / norm(u)^2) * u

        predator.vel = proj_u_v

    else
        



        # impact of sheep attraction on distance to sheep
        predator.vel = ((predator.pos - prey.pos)/current_distance)/dt
    
    end

    move_agent!(predator, model, dt)

    return
end