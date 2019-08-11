class RobotMarker {
  constructor(robotName, color, size=1) {
    this.name = robotName;
    this.color = color;
    this.size = size;
    this.marker_width = size;
    this.marker_height = size;
    this.graphic = null;
    this.scalingFactor = 0.1;
  }

  updateScaling(scaleX, scaleY) {
    this.marker_width = this.size/1.5;
    this.marker_height = this.size/1.5;
    let graphic = new createjs.Graphics();
    graphic.setStrokeStyle(1);
    graphic.beginFill(createjs.Graphics.getRGB(255,0,0));
    graphic.drawEllipse(-this.marker_width/2,-this.marker_height/2,this.marker_width,this.marker_height);
    this.marker = new createjs.Shape(graphic);
    this.markerText = new createjs.Text(this.name, '12px Arial', this.color);
    let bounds = this.markerText.getBounds();
    this.markerText.setTransform(
      this.markerText.x-this.scalingFactor*bounds.width/2,
      this.markerText.y-this.scalingFactor*bounds.height,
      this.scalingFactor,
      this.scalingFactor
    );
  }

  setLocation(x, y, theta=0) {
    this.marker.x = x;
    this.marker.y = y;
    this.marker.rotation = theta;
    let bounds = this.markerText.getTransformedBounds();
    this.markerText.setTransform(x-bounds.width/2, y-bounds.height, this.scalingFactor, this.scalingFactor);
  }

  addMarker(viewer) {
    this.updateScaling(viewer.scene.scaleX, viewer.scene.scaleY);
    viewer.addObject(this.marker);
    viewer.addObject(this.markerText);
  }
} 


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


function handleClick(evt) {
  console.log(evt);
  x_offset = window.gridClient.currentGrid.x;
  y_offset = window.gridClient.currentGrid.y;
  //console.log(gridClient.currentGrid);
  //console.log('Pixel Coordinates');
  //console.log(evt.stageX + ', ' + evt.stageY);
  //console.log('Scaled Coordinates');
  //console.log(evt.stageX/viewer.scene.scaleX + ', ' + evt.stageY/viewer.scene.scaleY);
  //console.log('Offset');
  //console.log(x_offset + ', ' + y_offset);
  x_click_map = evt.stageX/window.viewer.scene.scaleX+x_offset;
  y_click_map = evt.stageY/window.viewer.scene.scaleY+y_offset;
  //console.log('Transformed click coordinates');
  console.log(x_click_map + ', ' +  y_click_map);

  //s.x = x_click_map;
  //s.y = y_click_map;

  //nav_topic.publish(new ROSLIB.Message({
  //  header: {
  //    frame_id: 'map'
  //  },
  //  pose: {
  //    position: {
  //      x: x_click_map,
  //      y: -y_click_map,
  //      z: 0.0
  //    },
  //    orientation: heading_to_quaternion(0)
  //  }
  //}));
}


function start_robot_control(agent_name) {
  function func(mouse_event) {
    window.activeRobot = agent_name;
    // hide the robot select div
    document.getElementById("robot_select").style.display = "none";
    document.getElementById("robot_status").style.display = "flex";
    document.getElementById("command_row").style.display = "flex";
    // Create the main viewer.
    var viewer = new ROS2D.Viewer({
      divID : 'map',
      width : 600,
      height : 600
    });
    window.viewer = viewer;

    var status_srv = new ROSLIB.Service({
      ros : ros,
      name : '/task_server/get_agents_status',
      serviceType : '/long_term_deployment/AgentStatusList'
    });

    function query_status() {
      status_srv.callService({}, function(result) {
        for(let a of result.agent_statuses) {
          if(a.agent.agent_name == agent_name) {
            let div = document.getElementById("active");
            if(a.active_task.package_name !== "") {
              div.children[0].textContent = a.active_task.package_name + "/" + a.active_task.launchfile_name;
            } else {
              div.children[0].textContent = "No active task running...";
            }

            // clear out the old data
            let old_ul = document.getElementById("background");
            let ul = old_ul.cloneNode(false);
            old_ul.parentNode.replaceChild(ul, old_ul);

            for(let t of a.background_tasks) {
              let li = document.createElement("li");
              li.appendChild(document.createTextNode(t.package_name + "/" + t.launchfile_name));
              ul.appendChild(li);
            }
            if(a.background_tasks.length == 0) {
              let li = document.createElement("li");
              li.appendChild(document.createTextNode("No background tasks running..."));
              ul.appendChild(li);
            }
          }
        }
      });
    }

    setInterval(query_status, 1000);

    var nav_topic = new ROSLIB.Topic({
      ros : ros,
      name : '/move_base_simple/goal',
      messageType : 'geometry_msgs/PoseStamped'
    });
    window.nav_topic = nav_topic;

    viewer.scene.on("stagemousedown", handleClick);

    // Setup the map client.
    var gridClient = new ROS2D.OccupancyGridClient({
      ros : window.ros,
      rootObject : viewer.scene,
      topic: '/maps/graf/map',
      // Use this property in case of continuous updates
      continuous: true
    });
    window.gridClient = gridClient;
  
    // Scale the canvas to fit to the map
    gridClient.on('change', function() {
      console.log('Grid Client Change');
      viewer.scaleToDimensions(gridClient.currentGrid.width, gridClient.currentGrid.height);
      viewer.shift(gridClient.currentGrid.pose.position.x, gridClient.currentGrid.pose.position.y);
    });

    let marker = new RobotMarker(agent_name, 'red', 0.5);

    var robot_pose_topic = new ROSLIB.Topic({
      ros : ros,
      name : '/'+agent_name+'_agent/robot_pose',
      messageType : 'geometry_msgs/PoseStamped'
    });

    marker.addMarker(viewer);
    robot_pose_topic.subscribe(function(msg) {
      marker.setLocation(msg.pose.position.x, -msg.pose.position.y);
    });
  
  }
  return func;
}


function redraw_args(num_args) {
  let old_args = document.getElementById("arg_holder");
  let args = old_args.cloneNode(false);
  old_args.parentNode.replaceChild(args, old_args);
  console.log("Hello!");
  for(let i=0; i < num_args; i++) {
    let input = document.createElement("input");
    input.id = "task_arg"+(i+1);
    input.classList.add("form-control");
    input.classList.add("task_arg");
    input.placeholder = "(optional) argument #"+(i+1);
    args.appendChild(input);
  }
}

function init() {
  // setup some callbacks first
  let add_arg_btn = document.getElementById("add_args");
  let rem_arg_btn = document.getElementById("rem_args");
  let send_task_btn = document.getElementById("send_task");
  var num_args = 0;
  add_arg_btn.onclick = function(evt) {
    num_args++;
    redraw_args(num_args);
  }

  rem_arg_btn.onclick = function(evt) {
    num_args--;
    redraw_args(num_args);
    if(num_args < 0) {
      num_args = 0;
    }
  }

  send_task_btn.onclick = function(evt) {
    evt.preventDefault();

    console.log("sending task to " + window.activeRobot);

    let active_task_srv = new ROSLIB.Service({
      ros : ros,
      name : '/task_server/queue_task',
      serviceType : '/long_term_deployment/QueueTask'
    });

    let background_task_srv = new ROSLIB.Service({
      ros : ros,
      name : '/task_server/start_continuous_task',
      serviceType : '/long_term_deployment/QueueTask'
    });

    let agent_description = {
      agent_name: window.activeRobot,
      agent_type: window.activeRobot
    };

    var requested_task = new ROSLIB.ServiceRequest({
      task: {
        workspace_name : '',
        package_name : document.getElementById("pkgname").value,
        launchfile_name : document.getElementById("taskname").value,
        args : [],
        debug : false
      },
      agent: agent_description
    });

    // get the task arguments
    let elements = document.getElementById('arg_holder').childNodes;
    for(let i = 0; i < elements.length; i++) {
      requested_task.task.args.push(elements[i].value);
    }

    console.log(requested_task);
    console.log("number of args: " + elements.length);
    console.log(requested_task.task.args);

    let task_srv = null;
    if(document.getElementById("tasktype").value === "background") {
      task_srv = background_task_srv;
    } else {
      task_srv = active_task_srv;
    }

    task_srv.callService(requested_task, function(result) {
      console.log('Task Submission Result: '+ result.success);
    });
  }

  // Connect to ROS.
  var ros = new ROSLIB.Ros({
    url : 'ws://olorin.engr.oregonstate.edu:9090'
  });
  window.ros = ros;

  var agents_srv = new ROSLIB.Service({
    ros : ros,
    name : '/task_server/get_agents',
    serviceType : '/long_term_deployment/GetRegisteredAgents'
  });

  var robot_list = document.getElementById("robot_list");
  agents_srv.callService(new ROSLIB.ServiceRequest({}), function(result) {
    console.log(result);
    for(let a of result.agents) {
      let li = document.createElement("li");
      li.appendChild(document.createTextNode(a.agent_name));
      li.onclick = start_robot_control(a.agent_name);
      robot_list.appendChild(li);
    }
  });

}
