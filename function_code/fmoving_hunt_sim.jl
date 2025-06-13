using Pkg, InteractiveDynamics, CairoMakie, Agents, LinearAlgebra
using Random: Xoshiro

@agent struct Wolf(ContinuousAgent{2, Float64})
end

@agent struct Sheep(ContinuousAgent{2, Float64})
end

# Function to move a sheep at each time step
function animal_step!(agent::Sheep, model)

    center = model.center
    dt = model.dt
    sheep_tangent_acceleration = model.sheep_tangent_acceleration

    rotation_matrix = [cos(pi/2) sin(pi/2); -sin(pi/2) cos(pi/2)]
    radius = norm(agent.pos - center)
    unit_radial_vector = (center - agent.pos) / radius

    new_speed = norm(agent.vel) + sheep_tangent_acceleration * dt
    centrip_accel = (new_speed^2/radius)* unit_radial_vector

    agent.vel += centrip_accel * dt

    tangential_vector = rotation_matrix * unit_radial_vector
    proj = tangential_vector * dot(agent.vel, tangential_vector)/norm(tangential_vector)
    agent.vel = proj/norm(proj) * new_speed

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
    # wolf_encounter_speed = 1.0      # this is an arbitrary speed choice
    wolf_encounter_speed = model.wolf_encounter_speed
    min_safe_distance = model.min_safe_distance
    ww_force_coefficient = model.ww_force_coefficient
    dt = model.dt

    # for each neighbor wolf, find repulsive force α distance
    wolf_repulsion = [0, 0]
    for neighbor in allagents(model)

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

function initialize(total_agents, size, min_safe_distance, ww_force_coefficient, sw_force_coefficient,
                   wolf_encounter_speed, dt, seed,
                   center, sheep_tangent_acceleration, sheep_init_speed)
    space = ContinuousSpace(size; periodic = false)
    properties = Dict(
        :min_safe_distance => min_safe_distance,
        :ww_force_coefficient => ww_force_coefficient,
        :sw_force_coefficient => sw_force_coefficient,
        :wolf_encounter_speed => wolf_encounter_speed,
        :dt => dt,
        :center => center,
        :sheep_tangent_acceleration => sheep_tangent_acceleration
    )
    rng = Xoshiro(seed)

    model = StandardABM(Union{Wolf, Sheep}, space; properties = properties, agent_step! = animal_step!, rng,
        container = Vector, # agents are not removed, so we use this
        scheduler = Schedulers.Randomly() # all agents are activated once at random
    )

    rotation_matrix = [cos(pi/2) sin(pi/2); -sin(pi/2) cos(pi/2)]

    for n in 1:(total_agents - 1)
        add_agent!(Wolf, model; vel = (0.0, 0.0))
    end 

    rand_pos = [5 + 10 *rand(rng), 5 + 10*rand(rng)]
    radial_vector = rand_pos - center
    tangential_vector = rotation_matrix * radial_vector
    initial_vel = (tangential_vector/norm(tangential_vector)) * sheep_init_speed

    add_agent!(Sheep, model; pos = rand_pos, vel = initial_vel) # add one sheep
    return model
end

# attempt at a callable function to run the simulation
function fmoving_hunt_sim(min_safe_distance=1.0, ww_force_coefficient=0.5, sw_force_coefficient=2.0, 
                          wolf_encounter_speed=1.0, dt=0.25, 
                          total_agents=6, size=(20.0, 20.0), seed=125, framerate=15, frames=200,
                          center=[10.0, 10.0], sheep_tangent_acceleration=-0.1, sheep_init_speed=0.5)

    model = initialize(
        total_agents, size, min_safe_distance, ww_force_coefficient, sw_force_coefficient,
        wolf_encounter_speed, dt, seed,
        center, sheep_tangent_acceleration, sheep_init_speed
    )

   # Define color, size, and marker functions
    ac(a::Wolf) = :green
    ac(a::Sheep) = :blue
    as(a::Wolf) = 10
    as(a::Sheep) = 13
    am = 'o'

    # Create the animation
    abmvideo("wolf_hunt.mp4", model;
    title = "Moving Sheep Hunt Simulation", framerate, frames, ac, as, am)

end

# test with 5 wolves and 1 sheep
fmoving_hunt_sim(
    1.0,                    # min_safe_distance
    0.5,                    # ww_force_coefficient
    2.0,                    # sw_force_coefficient
    1.0,                    # wolf_encounter_speed (hardcoded inside animal_step! in 2nd version)
    0.25,                   # dt
    6,                      # total_agents (5 wolves + 1 sheep)
    (20.0, 20.0),           # size of the space
    125,                    # seed
    15,                     # framerate
    200,                    # frames
    [10.0, 10.0],           # center of circular sheep path
    -0.1,                   # sheep_tangent_acceleration
    0.5                     # sheep_init_speed
)