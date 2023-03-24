def get_params(d, h):
    assert ((d == 'lego') or (d == 'jaw')) 
    if d == 'lego':
        # RGBA
        return 4, h, (2,6) # channels, width
    elif d == 'jaw':
        # sRBA
        aspect_ratio = 275/331
        # angle of view
        near = 163 - 75 # camera is on a circle 163 from origin at (x,y,0)
        far = 163 + 75 # 50 is approx
        return 3, int(h*aspect_ratio), (near,far)