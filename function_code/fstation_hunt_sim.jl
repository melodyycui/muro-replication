using Pkg, InteractiveDynamics, CairoMakie, Agents, LinearAlgebra
using Random: Xoshiro

# Creating Animal agent type which consists of Sheep (group = 1) and Wolves (group = 2)
@agent struct Animal(ContinuousAgent{2, Float64})
    group::Int
end

# Function to move a wolf at each time step, assuming stationary prey (sheep)
function animal_step!(wolf, model)

    # identify the sheep in the model
    sheep = nothing
    for agent in allagents(model)
        if agent.group == 1
            sheep = agent
            break                       # exit loop once prey is found
        end
    end

    # Model the behavior of the wolf
    if wolf.group == 2              # check agent is a wolf
        
        current_distance = norm(sheep.pos - wolf.pos)

        # each neighbor wolf exerts repulsive force on current wolf agent
        wolf_repulsion = [0, 0] # stores net wolf-wolf repulsive force vector on current wolf
        for neighbor in allagents(model)

            if neighbor.id != wolf.id && neighbor.group == 2
            
                # sum up wolf-wolf repulsive force vectors
                # repulsive force is inversely proportional to distance between wolves
                distance_between = norm(wolf.pos - neighbor.pos)
                wolf_repulsion += model.ww_force_coefficient * (wolf.pos - neighbor.pos) / (distance_between)^2

            end

        end
    
        # once wolf is within a critical distance to the sheep, wolf will maintain that critical distance
        # and orbit around sheep
        if (current_distance <= model.min_safe_distance)

            # find direction of wolf to sheep, rotate 90 degrees to find tangential movement direction
            rotation_matrix = [cos(pi/2) sin(pi/2); -sin(pi/2) cos(pi/2)]
            u = rotation_matrix * (wolf.pos - sheep.pos)

            # projecting the repulsive force vector onto the tangential vector and multiply by wolf speed
            # to determine the velocity the wolf travels along the circle
            dot_product = dot(u, wolf_repulsion)
            proj_u_v = (dot_product / norm(u)^2) * u * model.wolf_encounter_speed
            wolf.vel = proj_u_v

        # if wolf is outside critical distance from sheep, wolf is attracted to sheep
        else

            # wolf moves in direction of sheep due to attractive force exerted by sheep on wolf
            wolf_chase_velocity = (sheep.pos - wolf.pos) / norm(sheep.pos - wolf.pos) * model.wolf_encounter_speed
            
            # wolf velocity impacted by both attraction to sheep and repulsion to neighboring wolves
            wolf.vel = wolf_chase_velocity + wolf_repulsion
    
        end
    
        # update model with new velocity after dt (timestep increment for simulation)
        move_agent!(wolf, model, model.dt)
    end
end

# this fn initializes our agent-based model for wolf hunt of single stationary prey
function initialize(total_agents, size, min_safe_distance, ww_force_coefficient, sw_force_coefficient, wolf_encounter_speed, dt, seed)
    space = ContinuousSpace(size; periodic = false)
    properties = Dict(:min_safe_distance => min_safe_distance, :ww_force_coefficient => ww_force_coefficient,
        :sw_force_coefficient => sw_force_coefficient, :wolf_encounter_speed => wolf_encounter_speed,
        :dt => dt
    )
    rng = Xoshiro(seed)

    model = StandardABM(Animal, space; properties, agent_step! = animal_step!, rng,
        container = Vector, # agents are not removed, so we use this
        scheduler = Schedulers.Randomly() # all agents are activated once at random
    )

    # adding wolf agents at random positions with velocity = 0
    for n in 1:(total_agents - 1)
        add_agent!(model; group = 2, vel = (0.0, 0.0))
    end 

    # adding a single sheep/prey agent at random position with velocity = 0
    add_agent!(model; group = 1, vel = (0.0, 0.0))
    return model
end

# this fn returns a video simulation of a wolf pack hunting a single stationary prey
function fstation_hunt_sim(min_safe_distance=2.0, ww_force_coefficient=0.5, sw_force_coefficient=2, 
                        wolf_encounter_speed=2.0, dt=0.05, 
                        total_agents=6, size=(20.0, 20.0), seed=125, framerate=15, frames=300)

    model = initialize(
        total_agents,
        size,
        min_safe_distance,
        ww_force_coefficient,
        sw_force_coefficient,
        wolf_encounter_speed,
        dt,
        seed
    )

   # Define color, size, and marker functions
    ac(a::Animal) = a.group == 1 ? :blue : :green
    as(a::Animal) = a.group == 1 ? 13 : 10
    am(a::Animal) = a.group == 1 ? :diamond : 'o'

    # Create the animation
    abmvideo("wolf_hunt.mp4", model;
    title = "Wolf Hunt Simulation", framerate, frames, ac, as, am)

end

# test with 5 wolves and 1 sheep
fstation_hunt_sim(2.0,                    # min_safe_distance
                  0.5,                    # ww_force_coefficient
                  2,                      # sw_force_coefficient
                  2.0,                    # wolf_encounter_speed
                  0.05,                   # dt
                  6,                      # total_agents
                  (20.0, 20.0),           # size of the space
                  125,                    # seed for random number generation
                  15,                     # framerate
                  300                     # frames for the video
                  )