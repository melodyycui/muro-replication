using Pkg, InteractiveDynamics, CairoMakie, Agents, LinearAlgebra
using Random: Xoshiro

# wolf agent type
@agent struct Wolf(ContinuousAgent{2, Float64})
end

# wolf agent type
@agent struct Sheep(ContinuousAgent{2, Float64})
end

# fn updates the position and velocity of a sheep agent after one time step, dt
function animal_step!(agent::Sheep, model)

    mass_sheep = 1

    # sheep agent wants to move away from wolf agents
    wolf_repulsion = [0, 0] # stores net repulsive force from wolves on sheep
    
    # each wolf exerts a repulsive force on the sheep agent
    for wolf in allagents(model)

        if isa(wolf, Wolf)

            # placeholder values
            a = 1
            b = 1
            
            # sum up the repulsive forces exerted by each wolf on the sheep
            distance_between = norm(agent.pos - wolf.pos)
            unit_vector = (agent.pos - wolf.pos) / (distance_between)
            wolf_repulsion += (a * exp(-b * distance_between)) * unit_vector

        end

    end

    # sheep-sheep repulsion, alignment, attraction

    n_att = 3
    n_ali = 2
    n_rep = 0
    sheep_attraction = [0, 0]
    sheep_repulsion = [0, 0]
    sheep_alignment = [0, 0]

    # need to replace
    drep = 0.5
    
    w_rep = 0.6
    w_att = 0.3
    w_ali = 0.2

    # sheep repulsed by nearby sheep that are too close
    for sheep in allagents(model)

        if isa(sheep, Sheep) && sheep.id != agent.id
            if norm(sheep.pos - agent.pos) < drep
                sheep_repulsion -= (w_rep)*(sheep.pos - agent.pos)/norm(sheep.pos - agent.pos)
                n_rep += 1
            end
        end
    end

    if (n_rep > 0)
        sheep_repulsion /= n_rep
    end

    neighbors = [i for i in nearby_agents(agent, model)]

    # sheep attracted to neighboring sheep
    # how to prevent repeated random selection?
    for i in 1:n_att

        if (length(neighbors) > 0)
            neighbor = neighbors[rand(1:length(neighbors))]
            sheep_attraction += (w_att/n_att) * (neighbor.pos - agent.pos)/norm(neighbor.pos - agent.pos)
        end

    end

    # sheep aligning with neighboring sheep
    for i in 1:n_ali

        if (length(neighbors) > 0)
            # ultimately have to make an array from the n_att sheep (rn, all of nearby sheep)
            neighbor = neighbors[rand(1:length(neighbors))]
            sheep_alignment += (w_ali/n_ali) * (neighbor.pos/norm(neighbor.pos))
        end
        
    end

    net_force = wolf_repulsion + sheep_repulsion + sheep_attraction + sheep_alignment
    agent.vel += (net_force/mass_sheep)*model.dt

    move_agent!(agent, model, model.dt)

end


# fn updates the position and velocity of a wolf agent after one time step, dt
function animal_step!(agent::Wolf, model)

    mass_wolf = 1
    ww_repulsion = [0, 0] # stores net wolf-wolf repulsive force vector on current wolf

    # each neighbor wolf exerts repulsive force on current wolf agent
    for neighbor in allagents(model)

        if neighbor.id != agent.id && isa(neighbor, Wolf)
            
            # sum up wolf-wolf repulsive force vectors
            distance_between = norm(agent.pos - neighbor.pos)
            ww_repulsion += model.ww_force_coefficient * (agent.pos - neighbor.pos) / (distance_between)^2

        end

    end

    ws_attraction = [0, 0]
    circling = false

    for sheep in allagents(model)

        if isa(sheep, Sheep)

            current_distance = norm(sheep.pos - agent.pos)

            # once wolf is within a critical distance to the sheep, wolf will maintain that critical distance
            # and orbit around sheep due to repulsion from other wolves
            if (current_distance <= model.min_safe_distance)

                # find direction of wolf to sheep, rotate 90 degrees to find tangential movement direction
                rotation_matrix = [cos(pi/2) sin(pi/2); -sin(pi/2) cos(pi/2)]
                u = rotation_matrix * (agent.pos - sheep.pos)

                # projecting the repulsive force vector onto the tangential vector and multiply by wolf speed
                # to determine the velocity the wolf travels along the circle
                dot_product = dot(u, ww_repulsion)
                proj_u_v = (dot_product / norm(u)^2) * u * model.wolf_chase_speed
                agent.vel = proj_u_v
                circling = true
                break

            else

                # placeholder values
                a = 500
                b = 0.8

                # summing attractive forces from sheep
                unit_vector = (sheep.pos - agent.pos) / current_distance
                ws_attraction += a * exp(-b * current_distance) * unit_vector
            end
        end
    end

    if (!circling)

        acceleration = (ww_repulsion + ws_attraction) / mass_wolf
        agent.vel += acceleration * model.dt
    
    end
    
    # update model with new velocity after dt (timestep increment for simulation)
    move_agent!(agent, model, model.dt)
end

# this fn initializes our agent-based model for wolf hunt of a single reactive/escaping prey
function initialize(; total_agents, size,
                    min_safe_distance, ww_force_coefficient,
                    sw_force_coefficient, wolf_chase_speed,
                    dt, seed, center, sheep_speed,)
    space = ContinuousSpace(size; periodic = false)

    properties = Dict{Symbol, Any}(
    :min_safe_distance => min_safe_distance,
    :ww_force_coefficient => ww_force_coefficient,
    :sw_force_coefficient => sw_force_coefficient,
    :wolf_chase_speed => wolf_chase_speed,
    :dt => dt,
    :center => center,
    :sheep_speed => sheep_speed, :in_range => false
)

    rng = Xoshiro(seed)

    model = StandardABM(Union{Wolf, Sheep}, space;
                        properties=properties,
                        agent_step! = animal_step!,
                        rng=rng,
                        container=Vector,
                        scheduler=Schedulers.Randomly())
    
    # placeholder parameters
    total_wolf = 6
    total_sheep = 10

    # adding wolf agents at random positions in top-left corner with velocity = 0
    for n in 1:(total_wolf)

        rand_pos = [2*rand(rng), 18 + 2*rand(rng)]
        add_agent!(Wolf, model; pos = rand_pos, vel = (0.0, 0.0))
    end

    # adding sheep agents around the center with velocity = 0
    for n in 1:(total_sheep)
        
        rand_pos = [5 + 10 *rand(rng), 5 + 10*rand(rng)]
        add_agent!(Sheep, model; pos = rand_pos, vel = (0.0, 0.0))

    end 

    return model
end

# this fn returns a video simulation of a wolf pack hunting a single reactive/escaping prey
function freactive_hunt_sim(min_safe_distance=1.0, ww_force_coefficient=0.5, sw_force_coefficient=2.0,
                             wolf_chase_speed=1.0, dt=0.25,
                             total_agents=6, size=(20.0,20.0), seed=125, framerate=15, frames=200,
                             center=[10.0,10.0], sheep_speed=0.5)

    model = initialize(total_agents=total_agents, size=size,
                       min_safe_distance=min_safe_distance,
                       ww_force_coefficient=ww_force_coefficient,
                       sw_force_coefficient=sw_force_coefficient,
                       wolf_chase_speed=wolf_chase_speed,
                       dt=dt, seed=seed, center=center,
                       sheep_speed=sheep_speed)

    # Define color, size, and marker functions
    ac(a::Wolf) = :green
    ac(a::Sheep) = :blue
    as(a::Wolf) = 10
    as(a::Sheep) = 13
    am(a::Wolf) = 'o'
    am(a::Sheep) = :diamond

    # Create the animation
    abmvideo("wolf_hunt.mp4", model;
    title = "Reactive Sheep Hunt Simulation", framerate, frames, ac, as, am)

end

# test with 3 wolves and 1 sheep
freactive_hunt_sim(
    1.0,                    # min_safe_distance
    0.01,                    # ww_force_coefficient
    1.0,                    # sw_force_coefficient
    1.0,                    # wolf_chase_speed
    0.25,                   # dt
    4,                      # total_agents 
    (20.0, 20.0),           # size of sim space
    124,                    # seed 
    1,                     # framerate
    11,                    # frames 
    [10.0, 10.0],           # center of sheep movement
    1.3                     # sheep_speed in m/s (found from Jadhav)
)