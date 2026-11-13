"""Fit camera parameters and triangulate drone positions from stereo observations.

Calibration uses reviewed observations and DJI telemetry from the calibration
interval. Camera parameters remain fixed during validation and testing.
Positions use local East/North coordinates and DJI-relative height.
"""
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import csv, io, json, hashlib, math, re, shutil, time, zipfile
import numpy as np
from scipy.optimize import least_squares
try:
    import cv2
except ImportError:
    cv2 = None


def read_csv(text):
    return list(csv.DictReader(io.StringIO(text.lstrip("\ufeff"))))

def json_write(path, data):
    def default(v):
        if isinstance(v, np.ndarray): return v.tolist()
        if isinstance(v, np.generic): return v.item()
        if isinstance(v, Path): return str(v)
        raise TypeError(type(v).__name__)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                    default=default, allow_nan=False), encoding="utf-8")

def csv_write(path, rows):
    if not rows: return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fields); writer.writeheader(); writer.writerows(rows)

def metadata_parameters(text, cfg):
    # Read camera metadata fields separately from video rotation tags.
    sections = {}
    for camera, title in [("right", "Right camera:"), ("left", "Left camera:")]:
        start = text.index(title) + len(title)
        ends = [text.find(t, start) for t in ["Right camera:", "Left camera:", "After syncing:"]]
        end = min([e for e in ends if e >= 0] or [len(text)])
        values = {}
        for line in text[start:end].splitlines():
            if ":" in line:
                key, val = line.split(":", 1); values[key.strip()] = val.strip()
        def dms(s):
            import re
            m = re.search(r'([\d.]+)\s+deg\s+([\d.]+)\x27\s+([\d.]+)"\s*([NSEW])', s)
            if not m: raise ValueError(f"Could not parse GPS field: {s}")
            a,b,c,h = m.groups()
            return (float(a)+float(b)/60+float(c)/3600)*(-1 if h in "SW" else 1)
        sections[camera] = {
            "latitude": dms(values["GPS Latitude"]),
            "longitude": dms(values["GPS Longitude"]),
            "altitude_m": float(values["GPS Altitude"].split()[0]),
            "horizontal_accuracy_tag_m": float(values["Location Accuracy Horizontal"]),
            "focal_35mm_equivalent_mm": float(values["Camera Focal Length 35mm Equivalent"]),
            "width": int(values["Image Width"]), "height": int(values["Image Height"]),
            "creation_date": values["Creation Date"], "filename": values["File Name"],
        }
    # Use a local tangent approximation for the initial GPS baseline check.
    left,right = sections["left"], sections["right"]
    lat = math.radians((left["latitude"]+right["latitude"])/2)
    dn = math.radians(right["latitude"]-left["latitude"])*6371008.8
    de = math.radians(right["longitude"]-left["longitude"])*6371008.8*math.cos(lat)
    du = right["altitude_m"]-left["altitude_m"]
    sections["gps_baseline_approx_m"] = math.sqrt(dn*dn+de*de+du*du)
    sections["scale_status"] = "coarse_phone_GPS_separation_not_surveyed_baseline"
    sections["synchronization"] = {
        "pairing_rule": "same_export_frame_index_as_candidate_only",
        "physical_synchronization_verified": False,
        "left_original_start_cut_s": cfg.left_original_start_cut_s,
        "right_original_start_cut_s": cfg.right_original_start_cut_s,
        "note": "Estimated residual time offsets assume a constant camera clock difference."
    }
    return sections

def make_K(meta, scale=1.):
    w,h = meta["width"],meta["height"]
    f = meta["focal_35mm_equivalent_mm"] * math.hypot(w,h) / math.hypot(36,24) * scale
    return np.array([[f,0,(w-1)/2],[0,f,(h-1)/2],[0,0,1.]], dtype=float)

def prepare_pairs(predictions, meta):
    groups = defaultdict(list)
    for row in predictions:
        camera = row["camera_id"]
        if camera not in ("left","right"): raise ValueError(f"Unknown camera: {camera}")
        key = (row["recording_id"], int(row["frame_index"]), camera)
        groups[key].append(row)
    pairs=[]
    for recording,frame in sorted({(r,f) for r,f,c in groups}):
        pair={"recording_id":recording,"frame_index":frame}
        for camera in ("left","right"):
            rows=groups.get((recording,frame,camera),[])
            pair[camera+"_detections"]=[r for r in rows if r.get("center_x_px", "") != ""]
            if not rows: pair[camera+"_filename"]=""; continue
            if len({r["filename"] for r in rows}) != 1:
                raise ValueError("Multiple filenames for the same camera and frame index")
            m=meta[camera]
            if any(int(r["width"])!=m["width"] or int(r["height"])!=m["height"] for r in rows):
                raise ValueError("Detection dimensions differ from camera metadata; check scaling")
            pair[camera+"_filename"]=rows[0]["filename"]
            pair[camera+"_elapsed_time_s"]=float(rows[0]["elapsed_time_s"])
        counts=[len(pair[c+"_detections"]) for c in ("left","right")]
        pair["pair_status"] = "missing_detection" if min(counts)==0 else (
            "multiple_detections_ambiguous" if max(counts)>1 else "single_detection_each_view")
        if all(c+"_elapsed_time_s" in pair for c in ("left","right")):
            pair["nominal_time_difference_ms"]=(pair["right_elapsed_time_s"]-pair["left_elapsed_time_s"])*1000
            if abs(pair["nominal_time_difference_ms"])>1:
                pair["pair_status"]="export_timestamps_mismatch"
        pairs.append(pair)
    if len({p["recording_id"] for p in pairs})!=1:
        raise ValueError("I process one recording per run; please split the inputs")
    return pairs

def normalize(points,K):
    return cv2.undistortPoints(np.asarray(points,float).reshape(-1,1,2),K,None).reshape(-1,2)

def triangulate(p,q,K1,K2,R,t):
    a,b=normalize(p,K1),normalize(q,K2)
    P1=np.column_stack((np.eye(3),np.zeros(3)))
    P2=np.column_stack((R,np.asarray(t).reshape(3)))
    H=cv2.triangulatePoints(P1,P2,a.T,b.T)
    safe=np.abs(H[3])>1e-12
    X=np.full((len(a),3),np.nan); X[safe]=(H[:3,safe]/H[3,safe]).T
    Y=(R@X.T).T+np.asarray(t).reshape(3)
    rays1=np.column_stack((a,np.ones(len(a))))
    rays2=(R.T@np.column_stack((b,np.ones(len(b)))).T).T
    cosine=np.sum(rays1*rays2,axis=1)/(np.linalg.norm(rays1,axis=1)*np.linalg.norm(rays2,axis=1))
    angle=np.degrees(np.arccos(np.clip(cosine,-1,1)))
    # Reject nearly parallel rays; positive depth is checked later.
    angle=np.minimum(angle,180-angle)
    with np.errstate(divide="ignore",invalid="ignore"):
        u=(K1@X.T).T; u=u[:,:2]/u[:,2,None]
        v=(K2@Y.T).T; v=v[:,:2]/v[:,2,None]
        residual=np.maximum(np.linalg.norm(u-np.asarray(p),axis=1),np.linalg.norm(v-np.asarray(q),axis=1))
    positive=np.isfinite(X).all(axis=1)&(X[:,2]>0)&(Y[:,2]>0)
    return X,positive,angle,residual


@dataclass
class Config:
    output_root: str = "outputs"
    # Load observations and telemetry from external files.
    results_path: str | None = None
    srt_path: str | None = None
    metadata_path: str | None = None
    export_root: str | None = None  # Optional source images for diagnostic previews.
    input_root: str = "."
    left_original_start_cut_s: float = 0.0
    right_original_start_cut_s: float = 0.0
    calibration_end_s: float = 120.0
    validation_start_s: float = 125.0
    validation_end_s: float = 166.0
    offset_bounds_s: tuple = (-3., 2.)
    initial_offsets_s: tuple = (-2., -1., 0., 1.)
    initial_focal_factors: tuple = (.8, 1., 1.5, 2.)
    focal_factor_bounds: tuple = (.6, 2.2)
    telemetry_grid_s: float = .1
    robust_loss_px: float = 20.
    initialization_ransac_px: float = 50.
    max_nfev: int = 500
    min_calibration_points: int = 20
    min_trajectory_singular_ratio: float = .03
    max_interpolation_gap_s: float = 1.1
    max_nearest_time_error_s: float = .1
    max_stereo_reprojection_px: float = 8.
    min_ray_angle_deg: float = 1.
    # Set the geometric screening threshold for triangulated points.
    seed: int = 42


def load_inputs(cfg):
    if not cfg.results_path or not cfg.srt_path or not cfg.metadata_path:
        raise ValueError('Missing required paths: --results-path, --srt-path, and --metadata-path')
    b={}
    path=Path(cfg.results_path)
    for name in ('predictions_tiled_merged.csv', 'run_info.json'):
        if path.is_dir():
            b[name]=(path/name).read_text(encoding='utf-8-sig')
        else:
            with zipfile.ZipFile(path) as z:
                found=[n for n in z.namelist() if Path(n).name==name]
                if len(found)!=1:
                    raise ValueError(f'The results ZIP must contain exactly one {name}')
                b[name]=z.read(found[0]).decode('utf-8-sig')
    b['flight_telemetry.srt']=Path(cfg.srt_path).read_text(encoding='utf-8-sig')
    b['extractedVideoMetadata.txt']=Path(cfg.metadata_path).read_text(encoding='utf-8-sig')
    b['source_srt_name']=Path(cfg.srt_path).name
    return b


def parse_telemetry(text,grid_s=.1):
    records=[]; first=None
    for block in re.split(r'\n\s*\n',text.strip()):
        d=re.search(r'\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+',block)
        if not d: continue
        fields=[re.search(k+r':\s*([-+\d.]+)',block) for k in ('latitude','longitude','rel_alt')]
        if not all(fields): raise ValueError('SRT record is missing latitude, longitude, or rel_alt')
        now=datetime.fromisoformat(d.group())
        if first is None:first=now
        records.append([(now-first).total_seconds(),*[float(m[1]) for m in fields]])
    raw=np.asarray(records,float)
    if len(raw)<20 or not np.isfinite(raw).all() or np.any(np.diff(raw[:,0])<=0):
        raise ValueError('SRT times must increase and contain at least 20 valid records')
    if np.any(np.abs(raw[:,1])>90) or np.any(np.abs(raw[:,2])>180):
        raise ValueError('SRT latitude or longitude is outside the valid range')
    lat0,lon0=raw[0,1:3];radius=6371008.8
    world=np.column_stack(((raw[:,2]-lon0)*math.pi/180*radius*math.cos(math.radians(lat0)),
                            (raw[:,1]-lat0)*math.pi/180*radius,raw[:,3]))
    # Interpolate onto a regular grid to handle repeated SRT GPS readings.
    grid=np.r_[np.arange(0,raw[-1,0],grid_s),raw[-1,0]]
    smooth=np.column_stack([np.interp(grid,raw[:,0],world[:,i]) for i in range(3)])
    return {'first_wallclock':first,'raw':raw,'raw_world':world,'grid':grid,'world':smooth,
            'origin_latitude':float(lat0),'origin_longitude':float(lon0)}


def nominal_offsets(meta,telemetry):
    # Interpret camera creation dates and SRT timestamps in the same local time zone.
    offsets={}
    for cam,cut in [('left',meta['synchronization']['left_original_start_cut_s']),
                    ('right',meta['synchronization']['right_original_start_cut_s'])]:
        start=datetime.strptime(meta[cam]['creation_date'],'%Y:%m:%d %H:%M:%S%z')
        local=(start+timedelta(seconds=cut)).replace(tzinfo=None)
        offsets[cam]=(local-telemetry['first_wallclock']).total_seconds()
    return offsets


def interpolate_world(telemetry,srt_seconds):
    ts=np.atleast_1d(np.asarray(srt_seconds,float));grid=telemetry['grid']
    good=np.isfinite(ts)&(ts>=grid[0])&(ts<=grid[-1])
    out=np.full((len(ts),3),np.nan)
    for j in range(3):out[good,j]=np.interp(ts[good],grid,telemetry['world'][:,j])
    return out,good


def split_name(t,cfg):
    if t < cfg.calibration_end_s:
        return 'calibration'
    if t < cfg.validation_start_s:
        return 'buffer'
    if t < cfg.validation_end_s:
        return 'validation'
    return 'test'


def intrinsic(f,meta):
    return np.array([[f,0,(meta['width']-1)/2],[0,f,(meta['height']-1)/2],[0,0,1.]],float)


def unpack_model(v,meta):
    K=intrinsic(float(np.exp(v[6])),meta);R=cv2.Rodrigues(v[:3])[0];t=np.asarray(v[3:6])
    return K,R,t


def project_world(world,v,meta):
    K,R,t=unpack_model(v,meta)
    uv=cv2.projectPoints(world,np.asarray(v[:3]),t,K,None)[0].reshape(-1,2)
    depth=((R@world.T).T+t)[:,2]
    return uv,depth


def fit_camera(camera,pairs,telemetry,meta,nominal,cfg):
    usable=[p for p in pairs if p['pair_status']=='single_detection_each_view'
            and p[camera+'_elapsed_time_s']<cfg.calibration_end_s]
    ts=np.array([p[camera+'_elapsed_time_s'] for p in usable])
    xy=np.array([[float(p[camera+'_detections'][0][k]) for k in ('center_x_px','center_y_px')] for p in usable])
    lo,hi=cfg.offset_bounds_s
    valid=(nominal+ts+lo>=telemetry['grid'][0])&(nominal+ts+hi<=telemetry['grid'][-1])
    ts,xy=ts[valid],xy[valid];frames=np.array([p['frame_index'] for p in usable])[valid]
    if len(ts)<cfg.min_calibration_points:
        raise ValueError(f'{camera}: not enough usable calibration correspondences')
    nominal_world,_=interpolate_world(telemetry,nominal+ts)
    singular=np.linalg.svd(nominal_world-nominal_world.mean(0),compute_uv=False)
    if singular[-1]/singular[0]<cfg.min_trajectory_singular_ratio:
        raise ValueError(f'{camera}: the calibration trajectory needs more 3D variation')
    f0=make_K(meta)[0,0];fmin,fmax=[f0*x for x in cfg.focal_factor_bounds]
    lower=np.r_[np.full(6,-np.inf),np.log(fmin),lo]
    upper=np.r_[np.full(6,np.inf),np.log(fmax),hi]
    def residual(v):
        world,ok=interpolate_world(telemetry,nominal+ts+v[7])
        if not ok.all():raise ValueError('An optimization candidate is outside SRT coverage')
        uv,_=project_world(world,v,meta)
        return (uv-xy).ravel()
    solutions=[];starts=[]
    for factor in cfg.initial_focal_factors:
        f=f0*factor
        if not fmin<f<fmax:continue
        K=intrinsic(f,meta)
        for offset in cfg.initial_offsets_s:
            if not lo<offset<hi:continue
            world,_=interpolate_world(telemetry,nominal+ts+offset)
            cv2.setRNGSeed(cfg.seed)
            ok,r,t,ii=cv2.solvePnPRansac(world,xy,K,None,iterationsCount=800,
                reprojectionError=cfg.initialization_ransac_px,confidence=.999,flags=cv2.SOLVEPNP_EPNP)
            if not ok:continue
            initial=np.r_[r.ravel(),t.ravel(),np.log(f),offset]
            opt=least_squares(residual,initial,bounds=(lower,upper),loss='soft_l1',
                f_scale=cfg.robust_loss_px,max_nfev=cfg.max_nfev,x_scale='jac',diff_step=1e-5)
            world,_=interpolate_world(telemetry,nominal+ts+opt.x[7])
            uv,depth=project_world(world,opt.x,meta)
            error=np.linalg.norm(uv-xy,axis=1)
            entry={'initial_focal_factor':factor,'initial_offset_s':offset,
                   'cost_on_calibration_only':float(opt.cost),'optimizer_success':bool(opt.success),
                   'final_focal_px':float(np.exp(opt.x[6])),'final_offset_s':float(opt.x[7]),
                   'calibration_positive_depth_fraction':float(np.mean(depth>0)),
                   'calibration_median_reprojection_px':float(np.median(error))}
            starts.append(entry)
            if opt.success and np.mean(depth>0)>=.95 and np.isfinite(opt.x).all():
                solutions.append((float(opt.cost),opt.x.copy()))
    if not solutions:raise ValueError(f'{camera}: no valid positive-depth camera fit was found')
    solutions.sort(key=lambda item:item[0]);cost,v=solutions[0]
    K,R,t=unpack_model(v,meta)
    warnings=[]
    if min(v[7]-lo,hi-v[7])<.05:warnings.append('time_offset_near_search_boundary')
    if min(np.exp(v[6])-fmin,fmax-np.exp(v[6]))<10:warnings.append('focal_length_near_search_boundary')
    model={'camera_id':camera,'K':K,'R_world_to_camera':R,'t_world_to_camera':t,
           'camera_center_world':-R.T@t,'params':v,'focal_px':float(np.exp(v[6])),
           'nominal_export_to_srt_s':nominal,'fitted_clock_correction_s':float(v[7]),
           'export_to_srt_s':float(nominal+v[7]),'distortion_coefficients_assumed':[0.,0.,0.,0.,0.],
           'principal_point_status':'fixed_at_image_centre_not_calibrated',
           'time_scale':1.,'time_scale_status':'fixed_no_speed_change_or_drift_estimated',
           'calibration_frames':frames.tolist(),'calibration_point_count':len(ts),
           'trajectory_singular_values':singular,'warnings':warnings,
           'model_selection':'minimum robust cost on calibration data only',
           'initializations':starts}
    print(f"{camera}: f={model['focal_px']:.1f}px; clock correction={v[7]:+.4f}s; calibration n={len(ts)}",flush=True)
    return model


def projection_diagnostics(pairs,models,telemetry,meta,cfg):
    rows=[]
    for pair in pairs:
        for camera in ('left','right'):
            ds=pair[camera+'_detections']
            if len(ds)!=1:continue
            model=models[camera];t=pair[camera+'_elapsed_time_s'];srt_t=t+model['export_to_srt_s']
            if split_name(t,cfg)=='buffer':continue
            world,ok=interpolate_world(telemetry,[srt_t])
            if not ok[0]:continue
            uv,depth=project_world(world,model['params'],meta[camera])
            det=ds[0];actual=np.array([float(det['center_x_px']),float(det['center_y_px'])])
            rows.append({'camera_id':camera,'frame_index':pair['frame_index'],'export_time_s':t,
                         'split':split_name(t,cfg),'srt_time_s':float(srt_t),
                         'observed_x_px':actual[0],'observed_y_px':actual[1],
                         'projected_srt_x_px':uv[0,0],'projected_srt_y_px':uv[0,1],
                         'reprojection_px':float(np.linalg.norm(uv[0]-actual)),
                         'positive_depth':bool(depth[0]>0)})
    return rows


def right_observation_at(target, track, cfg, required_split):
    """Match right-view observations by corrected time within the current split."""
    # Keep interpolation within the calibration, validation, or test interval.
    candidates = [r for r in track if r['split'] == required_split]
    if not candidates:
        return None

    times = np.array([r['time'] for r in candidates])
    j = int(np.searchsorted(times, target))
    if 0 < j < len(candidates):
        before, after = candidates[j - 1], candidates[j]
        gap = after['time'] - before['time']
        if 0 < gap <= cfg.max_interpolation_gap_s:
            weight = (target - before['time']) / gap
            uv = (1 - weight) * before['xy'] + weight * after['xy']
            return {
                'xy': uv,
                'method': 'linear_interpolation_of_two_observed_detections',
                'time_residual_s': 0.0,
                'source_frames': [before['frame'], after['frame']],
                'weight': float(weight),
                'source_gap_s': float(gap),
            }

    # Use the two neighboring timestamps within the current split.
    nearby = candidates[max(0, j - 1):min(len(candidates), j + 1)]
    nearest = min(nearby, key=lambda r: abs(r['time'] - target))
    residual = nearest['time'] - target
    if abs(residual) <= cfg.max_nearest_time_error_s:
        return {
            'xy': nearest['xy'],
            'method': 'nearest_detection_with_residual_time_error',
            'time_residual_s': float(residual),
            'source_frames': [nearest['frame']],
            'weight': 0.0,
            'source_gap_s': 0.0,
        }
    return None


def localize(pairs,models,telemetry,cfg):
    left,right=models['left'],models['right'];kl,kr=left['K'],right['K']
    rl,rr=left['R_world_to_camera'],right['R_world_to_camera']
    tl,tr=left['t_world_to_camera'],right['t_world_to_camera']
    R=rr@rl.T;t=tr-R@tl
    track=[]
    for p in pairs:
        dets=p['right_detections']
        if len(dets)==1 and split_name(p['right_elapsed_time_s'],cfg)!='buffer':
            d=dets[0];ts=p['right_elapsed_time_s']
            track.append({'time':ts,'xy':np.array([float(d['center_x_px']),float(d['center_y_px'])]),
                          'frame':p['frame_index'],'split':split_name(ts,cfg)})
    track.sort(key=lambda x:x['time']);rows=[]
    for pair in pairs:
        row={k:v for k,v in pair.items() if not k.endswith('_detections')}
        ts=pair.get('left_elapsed_time_s',pair.get('right_elapsed_time_s',0.))
        row.update(split=split_name(ts,cfg),status=pair['pair_status'],
                   left_count=len(pair['left_detections']),right_count=len(pair['right_detections']),
                   physical_synchronization_verified=False,ground_truth_accuracy_verified=False,
                   coordinate_system='local_East_North_and_DJI_relative_height')
        if row['split']=='buffer':
            row['status']='excluded_buffer_interval'
            rows.append(row)
            continue
        # Select a unique left detection. Match the right observation by corrected
        # time, which may point to a different source frame.
        left_count = len(pair['left_detections'])
        if left_count == 0:
            row['status'] = 'missing_left_detection'
            rows.append(row)
            continue
        if left_count > 1:
            row['status'] = 'multiple_left_detections_ambiguous'
            rows.append(row)
            continue
        srt_t=ts+left['export_to_srt_s'];right_t=srt_t-right['export_to_srt_s']
        row.update(target_srt_time_s=float(srt_t),target_right_export_time_s=float(right_t))
        obs=right_observation_at(right_t,track,cfg,row['split'])
        if obs is None:row['status']='no_close_time_aligned_right_detection';rows.append(row);continue
        dl=pair['left_detections'][0]
        p=np.array([[float(dl['center_x_px']),float(dl['center_y_px'])]])
        q=np.asarray(obs['xy']).reshape(1,2)
        row.update(left_x_px=p[0,0],left_y_px=p[0,1],aligned_right_x_px=q[0,0],aligned_right_y_px=q[0,1],
                   right_alignment_method=obs['method'],right_time_residual_s=obs['time_residual_s'],
                   right_source_frame_indices=';'.join(map(str,obs['source_frames'])),
                   right_interpolation_weight=obs['weight'],right_source_gap_s=obs['source_gap_s'])
        X,pos,angle,err=triangulate(p,q,kl,kr,R,t)
        row.update(ray_angle_deg=float(angle[0]),stereo_reprojection_max_px=float(err[0]) if np.isfinite(err[0]) else '')
        if not pos[0]:row['status']='rejected_negative_or_invalid_depth'
        elif angle[0]<cfg.min_ray_angle_deg:row['status']='rejected_low_ray_angle'
        else:
            world=(rl.T@(X[0]-tl)).ravel()
            row.update(estimate_east_m=float(world[0]),estimate_north_m=float(world[1]),
                       estimate_dji_relative_height_m=float(world[2]))
            # Keep finite hypotheses for diagnostics and screen accepted estimates.
            row['status']='gps_calibrated_exploratory' if err[0]<=cfg.max_stereo_reprojection_px else 'stereo_reprojection_warning'
            label,ok=interpolate_world(telemetry,[srt_t])
            if ok[0]:
                difference=world-label[0]
                row.update(recorded_east_m=float(label[0,0]),recorded_north_m=float(label[0,1]),
                           recorded_dji_relative_height_m=float(label[0,2]),
                           label_difference_3d_m=float(np.linalg.norm(difference)),
                           label_difference_horizontal_m=float(np.linalg.norm(difference[:2])),
                           label_difference_height_m=float(difference[2]))
            else:row['status']='outside_telemetry_coverage'
        rows.append(row)
    return rows,float(np.linalg.norm(t))


def evaluation_summary(rows):
    report={}
    for part in ('calibration','validation','test'):
        group=[r for r in rows if r['split']==part]
        all_finite=[r for r in group if 'label_difference_3d_m' in r]
        screened=[r for r in all_finite if r['status']=='gps_calibrated_exploratory']
        item={'total_frame_pairs':len(group),
              'single_detection_each_view':sum(r['pair_status']=='single_detection_each_view' for r in group),
              'time_aligned_right_observations':sum('right_alignment_method' in r for r in group),
              'positive_depth_estimates':len(all_finite),'screened_estimates':len(screened),
              'status_counts':dict(Counter(r['status'] for r in group))}
        # Report finite estimates and the geometry-screened subset.
        for name,records in [('all_finite',all_finite),('screened',screened)]:
            if records:
                d=np.array([r['label_difference_3d_m'] for r in records])
                item[name]={'count':len(d),'median_label_difference_m':float(np.median(d)),
                            'mean_label_difference_m':float(np.mean(d)),
                            'rmse_label_difference_m':float(np.sqrt(np.mean(d*d))),
                            'p90_label_difference_m':float(np.percentile(d,90)),
                            'median_stereo_reprojection_px':float(np.median([r['stereo_reprojection_max_px'] for r in records]))}
        report[part]=item
    report['excluded_buffer']={'total_frame_pairs':sum(r['split']=='buffer' for r in rows)}
    return report


def save_plots(rows,projections,cfg,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    finite=[r for r in rows if 'label_difference_3d_m' in r]
    if finite:
        fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True)
        names=[('east','East (m)'),('north','North (m)'),('dji_relative_height','DJI relative height (m)')]
        ts=[r['left_elapsed_time_s'] for r in finite]
        for ax,(name,title) in zip(axes,names):
            ax.scatter(ts,[r['estimate_'+name+'_m'] for r in finite],s=15,label='Stereo estimate',color='#1768ac')
            ax.scatter(ts,[r['recorded_'+name+'_m'] for r in finite],s=12,label='Aligned SRT reference',color='#d26b16',marker='x')
            ax.axvline(cfg.calibration_end_s,color='grey',ls='--');ax.axvline(cfg.validation_end_s,color='grey',ls='--')
            ax.set_ylabel(title);ax.grid(alpha=.2)
        axes[0].legend();axes[-1].set_xlabel('Left export time (s)')
        axes[0].set_title('GPS-assisted prototype | calibration / validation / test | reference is noisy telemetry')
        fig.tight_layout();fig.savefig(out/'trajectory_vs_srt.png',dpi=150);plt.close(fig)
        fig,ax=plt.subplots(figsize=(10,4))
        colors={'calibration':'#777777','validation':'#1768ac','test':'#b54836'}
        for part,col in colors.items():
            group=[r for r in finite if r['split']==part]
            ax.scatter([r['left_elapsed_time_s'] for r in group],[r['label_difference_3d_m'] for r in group],s=18,color=col,label=part)
        ax.set(xlabel='Left export time (s)',ylabel='Difference from SRT reference (m)',title='All finite positive-depth estimates; not ground-truth accuracy')
        ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'label_difference_by_time.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4))
    for camera,col in [('left','#1768ac'),('right','#d26b16')]:
        group=[r for r in projections if r['camera_id']==camera]
        ax.scatter([r['export_time_s'] for r in group],[r['reprojection_px'] for r in group],s=10,label=camera,color=col)
    ax.axvline(cfg.calibration_end_s,color='grey',ls='--');ax.axvline(cfg.validation_end_s,color='grey',ls='--')
    ax.set(xlabel='Export time (s)',ylabel='SRT-to-image residual (px)',title='Camera fit diagnostics: includes GPS, timing and detector errors')
    ax.legend();fig.tight_layout();fig.savefig(out/'camera_fit_residuals.png',dpi=150);plt.close(fig)


def optional_image_previews(pairs,rows,cfg,run_info,out):
    if cfg.export_root:roots=[Path(cfg.export_root)]
    else:roots=sorted({p.parent.parent for p in Path(cfg.input_root).rglob('frames.csv')
                       if p.parent.name=='left' and (p.parent.parent/'right/frames.csv').exists()})
    roots=[root for root in roots if all((root/c/'frames.csv').exists() and
             (not run_info.get('manifest_sha256',{}).get(c) or hashlib.sha256((root/c/'frames.csv').read_bytes()).hexdigest()==run_info['manifest_sha256'][c])
             for c in ('left','right'))]
    if len(roots)!=1:return 'raw_images_not_found_or_ambiguous; coordinates_and_plots_still_exported'
    root=roots[0];eligible=[i for i,r in enumerate(rows) if 'estimate_east_m' in r]
    for j in np.unique(np.linspace(0,len(eligible)-1,min(6,len(eligible)),dtype=int)) if eligible else []:
        i=eligible[j];pair=pairs[i];r=rows[i];panels=[]
        for camera in ('left','right'):
            im=cv2.imread(str(root/camera/'images'/pair[camera+'_filename']))
            if im is None:break
            # Show the nominal detection when available. Temporal matching may
            # select a right-side observation from another source frame.
            detections = pair[camera + '_detections']
            if len(detections) == 1:
                d = detections[0]
                x, y = int(float(d['center_x_px'])), int(float(d['center_y_px']))
                cv2.drawMarker(im, (x, y), (0, 255, 255), cv2.MARKER_CROSS, 18, 2)
            if camera == 'right' and 'aligned_right_x_px' in r:
                cv2.drawMarker(im, (int(r['aligned_right_x_px']),
                                    int(r['aligned_right_y_px'])),
                               (255, 255, 0), cv2.MARKER_CROSS, 18, 2)
            panel=cv2.resize(im,(960,540));cv2.rectangle(panel,(0,0),(960,36),(0,0,0),-1)
            cv2.putText(panel,f"{camera.upper()} | source frame {pair['frame_index']} | {pair[camera+'_elapsed_time_s']:.2f}s",(10,25),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1,cv2.LINE_AA)
            panels.append(panel)
        if len(panels)!=2:continue
        canvas=cv2.copyMakeBorder(np.hstack(panels),108,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
        lines=[f"{r['split'].upper()} | {r['status']} | estimated E {r['estimate_east_m']:.2f}m, N {r['estimate_north_m']:.2f}m, DJI relative height {r['estimate_dji_relative_height_m']:.2f}m",
               f"SRT reference E {r.get('recorded_east_m',0):.2f}m, N {r.get('recorded_north_m',0):.2f}m, relative height {r.get('recorded_dji_relative_height_m',0):.2f}m | label difference {r.get('label_difference_3d_m',0):.2f}m",
               'Yellow: detected center. Cyan: time-aligned right coordinate overlay.']
        for k,line in enumerate(lines):cv2.putText(canvas,line,(12,27+32*k),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,0),1,cv2.LINE_AA)
        cv2.imwrite(str(out/f"gps_localization_pair_f{pair['frame_index']:07d}.jpg"),canvas)
    return 'previews_created_from_original_export_images'


def run(cfg=Config()):
    if cv2 is None:
        raise ImportError('OpenCV is required for PnP and triangulation; install requirements.txt')
    started=time.perf_counter();cv2.setNumThreads(2)
    b=load_inputs(cfg);meta=metadata_parameters(b['extractedVideoMetadata.txt'],cfg)
    pairs=prepare_pairs(read_csv(b['predictions_tiled_merged.csv']),meta)
    telemetry=parse_telemetry(b['flight_telemetry.srt'],cfg.telemetry_grid_s)
    nominal=nominal_offsets(meta,telemetry)
    if not 0<cfg.calibration_end_s<cfg.validation_start_s<cfg.validation_end_s<max(p['left_elapsed_time_s'] for p in pairs):
        raise ValueError('The calibration, buffer, validation, and test ranges must be ordered')
    out=Path(cfg.output_root)/('flight01_gps_localization_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True,exist_ok=False)
    print(f"Loaded {len(pairs)} frame pairs: calibration <{cfg.calibration_end_s}s, "
          f"buffer <{cfg.validation_start_s}s, validation <{cfg.validation_end_s}s, then test.",flush=True)
    models={c:fit_camera(c,pairs,telemetry,meta[c],nominal[c],cfg) for c in ('left','right')}
    # Keep both camera models fixed for validation and test observations.
    json_write(out/'camera_parameters.json',models)
    projections=projection_diagnostics(pairs,models,telemetry,meta,cfg)
    rows,baseline=localize(pairs,models,telemetry,cfg)
    csv_write(out/'gps_localization.csv',rows);csv_write(out/'camera_projection_diagnostics.csv',projections)
    metrics=evaluation_summary(rows)
    alignment={'mapping':'srt_elapsed_seconds = export_elapsed_seconds + export_to_srt_s',
               'srt_first_wallclock':telemetry['first_wallclock'].isoformat(),
               'srt_wallclock_timezone':'assumed same local timezone as phone Creation Date',
               'left':{k:models['left'][k] for k in ['nominal_export_to_srt_s','fitted_clock_correction_s','export_to_srt_s']},
               'right':{k:models['right'][k] for k in ['nominal_export_to_srt_s','fitted_clock_correction_s','export_to_srt_s']},
               'same_physical_time_rule':'right_export_time = left_export_time + left_export_to_srt_s - right_export_to_srt_s',
               'right_minus_left_export_seconds_for_pairing':models['left']['export_to_srt_s']-models['right']['export_to_srt_s'],
               'physical_synchronization_verified':False,
               'telemetry_interpolation':'regular 0.1-second grid, then linear interpolation; no extrapolation',
               'remaining_uncertainty':'fitted offsets combine clock error, telemetry latency and model error; no drift or time warping fitted'}
    json_write(out/'time_alignment.json',alignment)
    save_plots(rows,projections,cfg,out)
    preview_status=optional_image_previews(pairs,rows,cfg,json.loads(b['run_info.json']),out)
    warnings=[w for m in models.values() for w in m['warnings']]
    for camera in ('left','right'):
        vals=[r['reprojection_px'] for r in projections if r['camera_id']==camera and r['split']=='validation']
        if vals and np.median(vals)>20:warnings.append(camera+'_validation_SRT_projection_residual_exceeds_20px')
    summary={'method':'GPS-assisted fixed-camera exploratory calibration and stereo triangulation',
             'status':'exploratory_coordinates_generated' if any('estimate_east_m' in r for r in rows) else 'no_valid_3d_estimates',
             'status_counts':dict(Counter(r['status'] for r in rows)),
             'metrics':metrics,'warnings':warnings,
             'origin_description':'first SRT position; exact coordinates are not exported here',
             'height_datum':'DJI relative-height reference; NOT altitude above sea level',
             'fitted_baseline_m':baseline,'phone_gps_baseline_approx_m_not_used_as_constraint':meta['gps_baseline_approx_m'],
             'ground_truth_accuracy_verified':False,'physical_synchronization_verified':False,
             'evaluation_scope':'within this flight, held-out contiguous time interval; not independent surveyed ground truth or a new flight',
             'test_used_for_fitting_or_model_selection':False,
             'config':asdict(cfg),'image_preview_status':preview_status,
             'total_wall_seconds':time.perf_counter()-started,
             'source_srt':b['source_srt_name'],
             'source_hashes':{k:hashlib.sha256(b[k].encode()).hexdigest() for k in ['predictions_tiled_merged.csv','flight_telemetry.srt','extractedVideoMetadata.txt']},
             'versions':{'opencv':cv2.__version__,'numpy':np.__version__}}
    json_write(out/'run_summary.json',summary)
    (out/'README.txt').write_text(
        'GPS-assisted camera calibration and stereo localization\n'
        '1. Camera parameters are fitted using the calibration interval (0 to <120 s).\n'
        '2. The buffer covers 120 to <125 s; validation is 125 to <166 s, and test starts at 166 s.\n'
        '3. estimate_* values in gps_localization.csv come from triangulation.\n'
        '4. recorded_* values come from the SRT telemetry reference.\n'
        '5. Coordinates use local East/North meters and DJI-relative height.\n'
        '6. Calibration fits focal length, camera pose, and constant time offset with a fixed principal point and zero distortion.\n'
        '7. Right-camera observations are matched by corrected time.\n'
        '8. Evaluation reports all finite estimates and the geometrically screened subset.\n'
        '9. The recorded values and estimated positions are compared within the same flight.\n',encoding='utf-8')
    archive=Path(shutil.make_archive(str(out),'zip',root_dir=out))
    print(json.dumps({'status':summary['status'],'counts':summary['status_counts'],
                      'validation':metrics['validation'],'test':metrics['test']},ensure_ascii=False,indent=2))
    print('Output directory:',out,'\nZIP:',archive)
    return out,archive,summary


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',default='outputs')
    parser.add_argument('--export-root')
    parser.add_argument('--input-root',default='.')
    parser.add_argument('--results-path',required=True,
                        help='Folder or ZIP with predictions_tiled_merged.csv and run_info.json')
    parser.add_argument('--srt-path',required=True,help='Path to an external drone telemetry SRT')
    parser.add_argument('--metadata-path',required=True,help='Path to external camera metadata text')
    parser.add_argument('--left-original-start-cut-s',type=float,default=0.)
    parser.add_argument('--right-original-start-cut-s',type=float,default=0.)
    parser.add_argument('--calibration-end-s',type=float,default=120.)
    parser.add_argument('--validation-start-s',type=float,default=125.)
    parser.add_argument('--validation-end-s',type=float,default=166.)
    args=parser.parse_args();run(Config(**vars(args)))
