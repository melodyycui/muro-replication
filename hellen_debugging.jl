using Pkg
Pkg.add("Agents")
Pkg.add("LinearAlgebra")
Pkg.add("InteractiveDynamics")
Pkg.add("CairoMakie")

using InteractiveDynamics
using CairoMakie

using Agents
using LinearAlgebra

using Random: Xoshiro # access the RNG object

#= NEED TO CONSIDER
1. Consistent naming of sheep/prey and wolf/predator
2. Double check vector directions (final - initial eg. positions)
3. need to calibrate magnitudes of things, such as size of grid, velocity (just in case vel too big, go out of bounds)
=#

# Maybe add health here too
@agent struct Animal(ContinuousAgent{2, Float64})
    group::Int # 1 for sheep, 2 for wolf
end

# will put in constructor; for now, putting hard-coded stuff here
min_safe_distance = 0.1
ww_force_coefficient = 0.5
sw_force_coefficient = 2 # force of sheep on wolf
dt = 1

function animal_step!(predator, model)

    prey = nothing # initialize prey variable
    for agent in allagents(model)
        if agent.group == 1 # identify the prey (sheep)
            prey = agent
            break # exit loop once prey is found
        end
    end

    if predator.group == 2 # wolf
        current_distance = norm(prey.pos - predator.pos)
        wolf_encounter_speed = 10 # this is an arbitrary speed choice

        # for each neighbor wolf, find repulsive force α distance
        wolf_repulsion = [0, 0]
        for neighbor in nearby_agents(predator, model)

            if neighbor.id != predator.id && neighbor.group == 2
            
                # sum up the repulsive force vectors to get acceleration
                distance_between = norm(predator.pos - neighbor.pos)
                wolf_repulsion += ww_force_coefficient * (predator.pos - neighbor.pos) / (distance_between)^2

            end

        end
    
        if (current_distance <= min_safe_distance)

            rotation_matrix = [cos(pi/2) sin(pi/2); -sin(pi/2) cos(pi/2)]

            # projecting the repulsive force vector onto the tangential vector
            # to determine the direction the wolf travels along the circle
            u = rotation_matrix * (predator.pos - prey.pos) # POTENTIAL FLAG: may be prey.pos - predator.pos (final - initial)
            dot_product = dot(u, wolf_repulsion)
            proj_u_v = (dot_product / norm(u)^2) * u * wolf_encounter_speed

            predator.vel = proj_u_v

        else # (wolf is outside critical distance)

            # impact of sheep attraction on distance to sheep
            wolf_encounter_velocity = (prey.pos - predator.pos) / norm(prey.pos - predator.pos) * wolf_encounter_speed

            predator.vel = wolf_encounter_velocity + wolf_repulsion ## THINK about how repulsive forces will impact the velocity 
    
        end

    move_agent!(predator, model, dt)
    return
end

function initialize(; total_agents = 5, size = (10.0, 10.0), min_safe_distance = 0.1, seed = 125)
    space = ContinuousSpace(size; periodic = false)
    properties = Dict(:min_safe_distance => min_safe_distance)
    rng = Xoshiro(seed)

    model = StandardABM(Animal, space; properties, agent_step! = animal_step!, rng,
        container = Vector, # agents are not removed, so we use this
        scheduler = Schedulers.Randomly() # all agents are activated once at random
    )

    for n in 1:(total_agents - 1)
        add_agent!(model; group = 2, vel = (0.0, 0.0))
    end 

    add_agent!(model; group = 1, vel = (0.0, 0.0)) # add one sheep
    return model
end

simulator = initialize()

# Attempting to Plot wolf model

model = initialize()
println("hello")

ac(a::Animal) = a.group == 1 ? :blue : :green
as(a::Animal) = a.group == 1 ? 13 : 10
am = 'o'

abmvideo("wolf_hunt.mp4", model;
title = "Wolf Hunt Simulation", framerate = 15, frames = 200, ac, as, am)

end