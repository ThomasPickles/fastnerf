def get_params(d, h):
    assert ((d == 'lego') or (d == 'jaw')) 
    if d == 'lego':
        # RGBA
        return 4, h # channels, width
    elif d == 'jaw':
        # sRBA
        aspect_ratio = 275/331
        return 3, int(h*aspect_ratio)