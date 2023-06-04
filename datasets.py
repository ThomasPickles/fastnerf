def get_params(d):
    assert ((d == 'jaw') or (d == 'walnut')) 
    if d == 'walnut':
        aspect_ratio = 2240./2368. 
        # angle of view
        # camera is on a circle 343 from origin at (x,y,0)
        scale = 349
        object_size = 39 # safer to overestimate
        return 1, scale, object_size, aspect_ratio
    elif d == 'jaw':
        # sRBA
        aspect_ratio = 275./331.
        # angle of view
        # camera is on a circle 163 from origin at (x,y,0)
        scale = 163
        object_size = 60 # safer to overestimate
        return 3, scale, object_size, aspect_ratio