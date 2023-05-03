def get_params(d, h):
    assert ((d == 'lego') or (d == 'jaw') or (d == 'walnut')) 
    if d == 'lego':
        # RGBA
        return 4, h, (2,6) # channels, width
    elif d == 'walnut':
        aspect_ratio = 2240/2368 
        # angle of view
        near = 343 - 150 # camera is on a circle 343 from origin at (x,y,0)
        far = 343 + 150 # 150 is approx
        return 1, int(h*aspect_ratio), (near,far)
    elif d == 'jaw':
        # sRBA
        aspect_ratio = 275/331
        # angle of view
        near = 163 - 60 # camera is on a circle 163 from origin at (x,y,0)
        far = 163 + 60 # 50 is approx
        return 3, int(h*aspect_ratio), (near,far)