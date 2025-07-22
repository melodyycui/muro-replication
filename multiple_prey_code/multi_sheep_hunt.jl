using Pkg, InteractiveDynamics, CairoMakie, Agents, LinearAlgebra
using Random: Xoshiro


# constructing sheep and wolf agent types:
@agent struct Wolf(ContinuousAgent{2, Float64})
end
@agent struct Sheep(ContinuousAgent{2, Float64})
end


# this fn updates the position and velocity of a sheep agent after one time step, dt:
function animal_step!(agent::Sheep, model)


   mass_sheep = 1


   # sheep agent wants to move away from wolf agents
   wolf_repulsion = [0, 0] # stores net repulsive force from wolves on sheep
  
   # each wolf exerts a repulsive force on the sheep agent
   for wolf in allagents(model)


       if isa(wolf, Wolf)


           # placeholder values
           a = 2.0
           b = 1
          
           # sum up the repulsive forces exerted by each wolf on the sheep
           dist_btwn_wolf_n_sheep = norm(agent.pos - wolf.pos)
           normalized_dist_btwn_wolf_n_sheep = (agent.pos - wolf.pos) / (dist_btwn_wolf_n_sheep)
           wolf_repulsion += (a * exp(-b * dist_btwn_wolf_n_sheep)) * normalized_dist_btwn_wolf_n_sheep


       end


   end


   # initialize num of sheep that exert attraction, alignment, and repulsion forces on current sheep agent:
   n_att = 3
   n_ali = 2
   n_rep = 0


   # initialize net sheep-sheep repulsion, alignment, and attraction forces:
   sheep_attraction = [0, 0]
   sheep_repulsion = [0, 0]
   sheep_alignment = [0, 0]


   # distance threshold for sheep-sheep repulsion
   drep = 1.0      # need to replace
  
   # weights for the attraction, alignment, and repulsion forces:
   w_att = 0.3
   w_ali = 0.2
   w_rep = 0.6


   # sheep that are too close to our current sheep agent exert a repulsive force on the agent:
   for sheep in allagents(model)


       if isa(sheep, Sheep) && sheep.id != agent.id


           if norm(sheep.pos - agent.pos) < drep


               sheep_repulsion -= (w_rep)*(sheep.pos - agent.pos)/norm(sheep.pos - agent.pos)
               n_rep += 1


           end


       end


   end


   # the repulsive force felt by current sheep agent is average of the repulsive sheep-sheep forces
   # exerted on it by all nearby sheep:
   if (n_rep > 0)
       sheep_repulsion /= n_rep
   end


   neighbors = [i for i in nearby_agents(agent, model)]


   # the current sheep agent is randomly attracted to one of its neighouring sheep:
   for _ in 1:n_att    # QUESTION: how to prevent repeated random selection?


       if (length(neighbors) > 0)


           neighbor = neighbors[rand(1:length(neighbors))]
           sheep_attraction += (w_att/n_att) * (neighbor.pos - agent.pos)/norm(neighbor.pos - agent.pos)


       end


   end


   # the current sheep agent will align with a select number of neighbouring sheep:
   for _ in 1:n_ali


       if (length(neighbors) > 0)


           # NOTE: ultimately have to make an array from the n_att sheep (rn, all of nearby sheep)
           neighbor = neighbors[rand(1:length(neighbors))] # POTENTIAL FLAG: sampling with replacement?
           sheep_alignment += (w_ali/n_ali) * (neighbor.pos/norm(neighbor.pos))


       end
      
   end


   # net force on the sheep agent is sum of the wolf-sheep repulsion, sheep-sheep repulsion,
   # sheep-sheep attraction, and sheep-sheep alignment forces acting on it:
   net_force = wolf_repulsion + sheep_repulsion + sheep_attraction + sheep_alignment


   # update the velocity of the sheep agent and move it:
   agent.vel += (net_force/mass_sheep)*model.dt
   move_agent!(agent, model, model.dt)


end


# this fn updates the position and velocity of a wolf agent after one time step, dt:
function animal_step!(agent::Wolf, model)


   mass_wolf = 1


   # Wolf–Wolf repulsion
   ww_repulsion = [0.0, 0.0]
   for neighbor in allagents(model)
       if neighbor.id != agent.id && isa(neighbor, Wolf)
           dist = norm(agent.pos - neighbor.pos)
           ww_repulsion += model.ww_force_coefficient * (agent.pos - neighbor.pos) / dist^2
       end
   end


   # Find closest sheep
   closest_sheep = nothing
   min_distance = Inf


  




   for sheep in allagents(model)
       if isa(sheep, Sheep)
           dist = norm(agent.pos - sheep.pos)
           if dist < min_distance
               min_distance = dist
               closest_sheep = sheep
           end
       end
   end


   if isnothing(closest_sheep)
       agent.vel = [0.0, 0.0]
       return
   end


   circling = false
   current_distance = min_distance


   # If the wolf is too close to the closest sheep, it will circle around it
   if current_distance <= model.min_safe_distance
       to_sheep = agent.pos - closest_sheep.pos
       rotation_matrix = [cos(pi/2) sin(pi/2); -sin(pi/2) cos(pi/2)]
       u = rotation_matrix * to_sheep


       dot_product = dot(u, ww_repulsion)
       proj_u_v = (dot_product / norm(u)^2) * u * model.wolf_chase_speed


       agent.vel = proj_u_v
       circling = true
   end


   # If not circling, chase the closest sheep
   if !circling
       a = 50
       b = 0.05
       unit_vec = (closest_sheep.pos - agent.pos) / current_distance
       ws_attraction = a * exp(-b * current_distance) * unit_vec


       acceleration = (ww_repulsion + ws_attraction) / mass_wolf
       agent.vel += acceleration * model.dt
   end


   move_agent!(agent, model, model.dt)
end


# this fn initializes our model with wolves randomly generated in the upper left quadrant
# and sheep randomly generated in the lower right quadrant:
function initialize(; size, min_safe_distance, ww_force_coefficient,
   sw_force_coefficient, wolf_chase_speed,
   dt, seed, center, sheep_speed, num_wolf, num_sheep)


   space = ContinuousSpace(size; periodic = false)


   properties = Dict(
       :min_safe_distance => min_safe_distance,
       :ww_force_coefficient => ww_force_coefficient,
       :sw_force_coefficient => sw_force_coefficient,
       :wolf_chase_speed => wolf_chase_speed,
       :dt => dt,
       :center => center,
       :sheep_speed => sheep_speed
   )


   rng = Xoshiro(seed)


   model = StandardABM(Union{Wolf, Sheep}, space;
       properties=properties,
       agent_step! = animal_step!,
       rng=rng,
       container=Vector,
       scheduler=Schedulers.Randomly())


   for _ in 1:num_wolf
       rand_pos = [5 + 10 * rand(rng), 40 + 5 * rand(rng)]  # x: 5–15, y: 40–45
       add_agent!(Wolf, model; pos = rand_pos, vel = (0.0, 0.0))
   end


   for _ in 1:num_sheep
       rand_pos = [15 + 20 * rand(rng), 15 + 20 * rand(rng)]  # x: 15–35, y: 15–35
       add_agent!(Sheep, model; pos = rand_pos, vel = (0.0, 0.0))
   end


   return model
end


# this fn make simulation video of the wolves hunting sheep:
function multi_hunt_sim(; min_safe_distance=1.0, ww_force_coefficient=1.0,
   sw_force_coefficient=2.0, wolf_chase_speed=1.0, dt=0.1,
   size=(20.0, 20.0), seed=125, framerate=15, frames=200,
   center=[10.0, 10.0], sheep_speed=0.5, num_wolf=3, num_sheep=5)


   model = initialize(
       size=size,
       min_safe_distance=min_safe_distance,
       ww_force_coefficient=ww_force_coefficient,
       sw_force_coefficient=sw_force_coefficient,
       wolf_chase_speed=wolf_chase_speed,
       dt=dt,
       seed=seed,
       center=center,
       sheep_speed=sheep_speed,
       num_wolf=num_wolf,
       num_sheep=num_sheep
   )


   ac(a::Wolf) = :green
   ac(a::Sheep) = :blue
   as(a::Wolf) = 10
   as(a::Sheep) = 13
   am(a::Wolf) = 'o'
   am(a::Sheep) = :diamond


   abmvideo("wolf_hunt.mp4", model;
       title="Multi Wolf–Sheep Physics Hunt",
       framerate=framerate,
       frames=frames,
       ac=ac, as=as, am=am)
end


# testing
multi_hunt_sim(
   min_safe_distance=5.0,
   ww_force_coefficient=1.0,
   sw_force_coefficient=5.0,
   wolf_chase_speed=1.0,
   dt=0.05,
   size=(50.0, 50.0),
   seed=124,
   framerate=15,
   frames=2000,
   center=[10.0, 10.0],
   sheep_speed=0.1,
   num_wolf=3,
   num_sheep=5
  
)
