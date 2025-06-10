using Agents, LinearAlgebra, Random


#= NEED TO CONSIDER
1. Consistent naming of sheep/prey and wolf/predator
2. Double check vector directions (final - initial eg. positions)
3. need to calibrate magnitudes of things, such as size of grid, velocity (just in case vel too big, go out of bounds)
=#

# Maybe add health here too?
@agent struct Wolf(ContinuousAgent{2, Float64})
#    group::Int
    vel::Vector{Float64}
end

@agent struct Sheep(ContinuousAgent{2, Float64})
#    group::Int
end

# will put in constructor; for now, putting hard-coded stuff here
min_safe_distance = 0.1
ww_force_coefficient = 0.5
sw_force_coefficient = 2 # force of sheep on wolf
dt = 1

# not sure if this function is relevant or needed
function add_agent_single(model)
    pos = rand(model.space)
    vel = zeros(2)
    return add_agent!(Wolf(pos, vel), model)
end

function wolf_step!(predator, prey, model)

    current_distance = norm(prey.pos - predator.pos)
    wolf_encounter_speed = 10 # this is an arbitrary speed choice

    # for each neighbor wolf, find repulsive force α distance
    wolf_repulsion = zeros(2)
    for neighbor in nearby_agents(predator, model)
        
        if neighbor.id == predator.id
            continue

        # sum up the repulsive force vectors to get acceleration
        else
            e = 1e-8   
            distance_between = max(norm(predator.pos - neighbor.pos), e)

            wolf_repulsion += ww_force_coefficient * (predator.pos - neighbor.pos) / (distance_between)^2
        end
    end
    
    if (current_distance <= min_safe_distance)

        rotation_matrix = [cos(pi/2) sin(pi/2); -sin(pi/2) cos(pi/2)]

        # projecting the repulsive force vector onto the tangential vector
        # to determine the direction the wolf travels along the circle
        u = rotation_matrix * (predator.pos - prey.pos)
        dot_product = dot(u, wolf_repulsion)
        proj_u_v = (dot_product / norm(u)^2) * u

        predator.vel = proj_u_v

    else

        # impact of sheep attraction on distance to sheep
        wolf_encounter_velocity = (prey.pos - predator.pos) / norm(prey.pos - predator.pos) * wolf_encounter_speed

        predator.vel = wolf_encounter_velocity + wolf_repulsion ## THINK about how repulsive forces will impact the velocity 
    
    end

    move_agent!(predator, model, dt)
    return

end

using Random: Xoshiro # access the RNG object

# predator.pos = clamp.(predator.pos, 0, model.space.extent)
function initialize(; total_agents = 5, size = (10.0, 10.0), min_safe_distance = 0.1, seed = 125)
    space = ContinuousSpace(size; periodic = false;)
    properties = Dict(:min_safe_distance => min_safe_distance)
    rng = Xoshiro(seed)
    model = StandardABM(Wolf, space; properties, agent_step! = wolf_step!, rng,
    container = Vector, # agents are not removed, so we use this
    scheduler = Schedulers.Randomly() # all agents are activated once at random
    )

    for n in 1:total_agents
        add_agent_single(model;)
    end 
    return model
end

simulator = initialize()

