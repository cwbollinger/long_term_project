
function heading_to_quaternion(yaw) {
  var roll = 0.0;
  var pitch = 0.0;
  var yaw = parseFloat(yaw);
  var phi = roll / 2.0;
  var the = pitch / 2.0;
  var psi = yaw / 2.0;
  var x = Math.sin(phi) * Math.cos(the) * Math.cos(psi) - Math.cos(phi) * Math.sin(the) * Math.sin(psi);
  var y = Math.cos(phi) * Math.sin(the) * Math.cos(psi) + Math.sin(phi) * Math.cos(the) * Math.sin(psi);
  var z = Math.cos(phi) * Math.cos(the) * Math.sin(psi) - Math.sin(phi) * Math.sin(the) * Math.cos(psi);
  var w = Math.cos(phi) * Math.cos(the) * Math.cos(psi) + Math.sin(phi) * Math.sin(the) * Math.sin(psi);
  
  orientation = new ROSLIB.Quaternion({
    x : x,
    y : y,
    z : z,
    w : w
  });
  orientation.normalize();
  return orientation;
}

function time_to_seconds(time_string) {
  return 3600*parseInt(time_string.slice(0,2)) + 60*parseInt(time_string.slice(3,5));
}

function init() {
  // Connect to ROS.
  var ros = new ROSLIB.Ros({
    url : 'ws://olorin.engr.oregonstate.edu:9090'
  });

  var task_srv = new ROSLIB.Service({
    ros : ros,
    name : '/central_planner/schedule_task',
    serviceType : '/long_term_deployment/ScheduleTask'
  });

  var requested_task = new ROSLIB.ServiceRequest({
    location : {
      position : {
        x : 0.0,
        y : 0.0,
        z : 0.0
      },
      orientation : heading_to_quaternion(0)
    },
    task : {
       workspace_name : '',
       package_name : 'long_term_deployment',
       launchfile_name : 'test_task',
       args : ['5.0'],
       debug : false,
    },
    start_after : {secs : 0, nsecs : 0},
    finish_before : {secs : 0, nsecs : 0},
    estimated_duration : {secs : 10, nsecs : 0},
  })

  document.getElementById("taskSubmit").addEventListener("click", function(e) {
    e.preventDefault();
    var base_time = new Date(document.getElementById("taskDate").value).getTime()/1000
    var start_after = document.getElementById("taskStartTime").value; 
    var end_before = document.getElementById("taskEndTime").value;

    requested_task.start_after.secs = time_to_seconds(start_after);
    requested_task.finish_before.secs = time_to_seconds(end_before);

    requested_task.location.position.x = parseFloat(document.getElementById("x").value);
    requested_task.location.position.y = parseFloat(document.getElementById("y").value);
    requested_task.location.orientation = heading_to_quaternion(document.getElementById("heading").value);

    requested_task.task.package_name = document.getElementById("taskPackage").value;
    requested_task.task.launchfile_name = document.getElementById("taskType").value;
    console.log(requested_task);
    task_srv.callService({task : requested_task}, function(result) {
      console.log('Task Submission Result: '+ result.success);
    });
  });
}
