using Pkg, InteractiveDynamics, CairoMakie, Agents, LinearAlgebra
using Random: Xoshiro

@agent struct Wolf(ContinuousAgent{2, Float64})
end

@agent struct Sheep(ContinuousAgent{2, Float64})
end

# can maybe be put in our slider/hard coded parameters
min_safe_distance = 1.0                 # critical distance at which wolf begins exhibiting encircling behavior
ww_force_coefficient = 0.5              # coefficient of repulsive force exerted by wolf on wolf
sw_force_coefficient = 2                # coefficient of repulsive force exerted by sheep on wolf
ws_repulsion = 1.0
sheep_tangent_acceleration = - 0.1      # constant tangential deceleration (slowing down)
dt = 0.25                               # time step for simulation
rotation_matrix = [cos(pi/2) sin(pi/2); -sin(pi/2) cos(pi/2)]
center = [10.0, 10.0]
sheep_speed = 0.1                  # initial tangential speed for sheep

# Function to move a sheep at each time step
function animal_step!(agent::Sheep, model)

    wolf_repulsion = [0, 0]
    for wolf in allagents(model)

        if wolf.id != agent.id
            
            # sum up the repulsive force vectors to get acceleration
            distance_between = norm(agent.pos - wolf.pos)
            wolf_repulsion += ws_repulsion * (agent.pos - wolf.pos) / (distance_between)^2

        end

    end

    agent.vel = wolf_repulsion / norm(wolf_repulsion) * sheep_speed
    move_agent!(agent, model, dt)

end


# Function to move a wolf at each time step
function animal_step!(agent::Wolf, model)

    # identify the sheep in the model
    sheep = nothing
    for a in allagents(model)
        if typeof(a) == Sheep
            sheep = a
            break                       # exit loop once prey is found
        end
    end

    # Model the behavior of the wolf
    current_distance = norm(sheep.pos - agent.pos)
    wolf_encounter_speed = 1.0      # this is an arbitrary speed choice

    # for each neighbor wolf, find repulsive force α distance
    wolf_repulsion = [0, 0]
    for neighbor in nearby_agents(agent, model)

        if neighbor.id != agent.id && isa(neighbor, Wolf)
            
            # sum up the repulsive force vectors to get acceleration
            distance_between = norm(agent.pos - neighbor.pos)
            wolf_repulsion += ww_force_coefficient * (agent.pos - neighbor.pos) / (distance_between)^2

        end

    end
    
    if (current_distance <= min_safe_distance)

        rotation_matrix = [cos(pi/2) sin(pi/2); -sin(pi/2) cos(pi/2)]

        # projecting the repulsive force vector onto the tangential vector
        # to determine the direction the wolf travels along the circle
        u = rotation_matrix * (agent.pos - sheep.pos)
        dot_product = dot(u, wolf_repulsion)
        proj_u_v = (dot_product / norm(u)^2) * u * wolf_encounter_speed

        agent.vel = proj_u_v

    else

        # impact of sheep attraction on distance to sheep
        wolf_encounter_velocity = (sheep.pos - agent.pos) / norm(sheep.pos - agent.pos) * wolf_encounter_speed

        agent.vel = wolf_encounter_velocity + wolf_repulsion ## THINK about how repulsive forces will impact the velocity 
    
    end
    
    move_agent!(agent, model, dt)
end

function initialize(; total_agents = 4, size = (20.0, 20.0), min_safe_distance = 0.1, seed = 124)
    space = ContinuousSpace(size; periodic = false)
    properties = Dict{Symbol, Any}(:min_safe_distance => min_safe_distance)

    rng = Xoshiro(seed)

    model = StandardABM(Union{Wolf, Sheep}, space; properties = properties, agent_step! = animal_step!, rng,
        container = Vector, # agents are not removed, so we use this
        scheduler = Schedulers.Randomly() # all agents are activated once at random
    )

    for n in 1:(total_agents - 1)
        add_agent!(Wolf, model; vel = (0.0, 0.0))
    end 

    rand_pos = [5 + 10 *rand(rng), 5 + 10*rand(rng)] # more central random init position so vid stays in frame


    add_agent!(Sheep, model; pos = rand_pos, vel = (0.0, 0.0)) # add one sheep
    return model
end



# Attempting to Plot wolf model

model = initialize()

# Prepare containers for wolf trajectories
n_wolves = count(a -> isa(a, Wolf), allagents(model))
all_wolf_x = [Float64[] for _ in 1:n_wolves]
all_wolf_y = [Float64[] for _ in 1:n_wolves]

# Prepare containers for sheep trajectories
n_sheep = count(a -> isa(a, Sheep), allagents(model))
all_sheep_x = [Float64[] for _ in 1:n_sheep]
all_sheep_y = [Float64[] for _ in 1:n_sheep]

function record_positions!(model)
    wolves = [a for a in allagents(model) if isa(a, Wolf)]
    for (i, wolf) in enumerate(wolves)
        push!(all_wolf_x[i], wolf.pos[1])
        push!(all_wolf_y[i], wolf.pos[2])
    end

    sheep = [a for a in allagents(model) if isa(a, Sheep)]
    for (i, sheep) in enumerate(sheep)
        push!(all_sheep_x[i], sheep.pos[1])
        push!(all_sheep_y[i], sheep.pos[2])
    end

end


# Run the simulation and record
steps = 50
for _ in 1:steps
    step!(model)
    record_positions!(model)
end

# Plot time series of wolf and sheep positions
function plot_all_trajectories(wolf_x, wolf_y, sheep_x, sheep_y)
    fig = Figure(resolution=(700, 700))
    ax = Axis(fig[1, 1], title="Wolf and Sheep Trajectories", xlabel="x", ylabel="y")

    # Plot wolves (multiple agents)
    for i in 1:length(wolf_x)
        scatter!(ax, wolf_x[i], wolf_y[i], label = "Wolf $i", color = RGBf(rand(3)...))
    end

    # Plot sheep (multiple agents)
    for i in 1:length(sheep_x)
        scatter!(ax, sheep_x[i], sheep_y[i], label = "Sheep $i", color = :blue)
    end

    axislegend(ax)
    fig
end

fig = plot_all_trajectories(all_wolf_x, all_wolf_y, all_sheep_x, all_sheep_y)
display(fig)


#= ac(a::Wolf) = :green
ac(a::Sheep) = :blue
as(a::Wolf) = 10
as(a::Sheep) = 13
am = 'o'
abmvideo("reactive_prey_hunt.mp4", model;
title = "Wolf Hunt Simulation", framerate = 15, frames = 200, ac, as, am) =#