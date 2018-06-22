// dimensions
var width = 800;
var height = 700;

var margin = {
    top: 10,
    bottom: 10,
    left: 10,
    right: 10,
}

// create an svg to draw in
var svg = d3.select("#jumbosvg")
    .append("svg")
    .attr("width", width)
    .attr("height", height)
    .append('g')
    .attr('transform', 'translate(' + margin.top + ',' + margin.left + ')');

width = width - margin.left - margin.right;
height = height - margin.top - margin.bottom;
var color = d3.scaleOrdinal(d3.schemeCategory10);

var simulation = d3.forceSimulation()
    // pull nodes together based on the links between them
    .force("link", d3.forceLink().id(function(d) {
        return d.id;
    })
    .strength(0.025))
    // push nodes apart to space them out
    .force("charge", d3.forceManyBody().strength(-1500))
    // add some collision detection so they don't overlap
    .force("collide", d3.forceCollide().radius(12))
    // and draw them around the centre of the space
    .force("center", d3.forceCenter(width/2 , height/2 ));

// load the graph
d3.json("datalabels.json", function(error, graph) {
    // set the nodes
    var nodes = graph.nodes;
    // links between nodes
    var links = graph.links;

    // add the curved links to our graphic
    var link = svg.selectAll(".link")
        .data(links)
        .enter()
        .append("path")
        .attr("class", "link")
        .style("stroke-width", function(d) { return Math.sqrt(d.value); })
        .attr('stroke', function(d){
            return "#ddd";
        });

    // add the nodes to the graphic
    var node = svg.selectAll(".node")
        .data(nodes)
        .enter().append("g")
        .call(d3.drag()
          .on("start", dragstarted)
          .on("drag", dragged)
          .on("end", dragended));

        function dragstarted(d) {
          if (!d3.event.active) simulation.alphaTarget(0.3).restart();
          d.fx = d.x;
          d.fy = d.y;
        }

        function dragged(d) {
          d.fx = d3.event.x;
          d.fy = d3.event.y;
        }

        function dragended(d) {
          if (!d3.event.active) simulation.alphaTarget(0);
          d.fx = null;
          d.fy = null;
        }

    // a circle to represent the node
    node.append("circle")
        .attr("class", "node")
        .attr("r", function(d) {
            return d.size+15;
        })
        .attr("fill", function(d) {
            return color(d.group);
        })
        .on("mouseover", mouseOver(.2))
        .on("mouseout", mouseOut)


    // hover text for the node
    node.append("title")
        .text(function(d) {
            return d.info[0][0];
        });


    // Variable to keep track of whether node has been clicked
    var tip;
    var tip2;

    // Remove text if already clicked
    node.on("click", function(){
            if (tip) tip.remove();
            if (tip2) tip2.remove();
          });

    // Action when clicked
    node.on("click", function(d){
            d3.event.stopPropagation();

             if (tip) tip.remove();


            tip  = svg.append("g")
              .attr("transform", "translate(" + d.x  + "," + d.y + ")");


            // Adds keywords close to nodes

            var i;
            for (i = 0; i < d.info.length; i++) {
            tip.append("text")
              .text(d.info[i].text)
              .attr("dy", (1.7*i+1)+"em")
              .attr("dx", 0.5*(1.5*i+2)+"em")
              .style('font-size',10*d.info[i].size+10+"px")
              .style("fill", color(d.group));

            }


            // Adds description top right corner
            tip2 = d3.select('body')
            .select('#infogroup')
            .text('This is group ' + d.group + ', whose keywords are');

            var html_to_add = []

            html_to_add.push('<h2>This is group ' + (d.group + 1) + ', whose most cited articles are </h2>')

            // var i;
            // for (i = 0; i < d.info.length; i++) {
            // html_to_add.push( d.info[i].text + ', ')
            //
            // }
            // html_to_add.push('</p> </br> <h3> The most cited articles in this group are </h3> <ol>')
            var url_test = "{% url 'app_compare:article_detail' slug=1678299 %}"
            var i;
            for (i = 0; i < d.infotopn.length; i++) {
               html_to_add.push(' <p> <a href= '+ url_test + ' >'  + d.infotopn[i].title + "</a> </p>");
                    };

             tip2.html(html_to_add.join(' '))




 });


    // add the nodes to the simulation and
    // tell it what to do on each tick
    simulation
        .nodes(nodes)
        .on("tick", ticked);

    // add the links to the simulation
    simulation
        .force("link")
        .links(links);

    // on each tick, update node and link positions
    function ticked() {
        link.attr("d", positionLink);
        node.attr("transform", positionNode);
    }

    // links are drawn as curved paths between nodes,
    // through the intermediate nodes
    function positionLink(d) {
        var offset = 30;

        var midpoint_x = (d.source.x + d.target.x) / 2;
        var midpoint_y = (d.source.y + d.target.y) / 2;

        var dx = (d.target.x - d.source.x);
        var dy = (d.target.y - d.source.y);

        var normalise = Math.sqrt((dx * dx) + (dy * dy));

        var offSetX = midpoint_x + offset*(dy/normalise);
        var offSetY = midpoint_y - offset*(dx/normalise);

        return "M" + d.source.x + "," + d.source.y +
            "S" + offSetX + "," + offSetY +
            " " + d.target.x + "," + d.target.y;
    }

    // move the node based on forces calculations
    function positionNode(d) {
        // keep the node within the boundaries of the svg
        if (d.x < 0) {
            d.x = 0
        };
        if (d.y < 0) {
            d.y = 0
        };
        if (d.x > width) {
            d.x = width
        };
        if (d.y > height) {
            d.y = height
        };
        return "translate(" + d.x + "," + d.y + ")";
    }

    // build a dictionary of nodes that are linked
    var linkedByIndex = {};
    links.forEach(function(d) {
        linkedByIndex[d.source.index + "," + d.target.index] = 1;
    });

    // check the dictionary to see if nodes are linked
    function isConnected(a, b) {
        return linkedByIndex[a.index + "," + b.index] || linkedByIndex[b.index + "," + a.index] || a.index == b.index;
    }

    // fade nodes on hover
    function mouseOver(opacity) {
        return function(d) {
            // check all other nodes to see if they're connected
            // to this one. if so, keep the opacity at 1, otherwise
            // fade
            node.style("stroke-opacity", function(o) {
                thisOpacity = isConnected(d, o) ? 1 : opacity;
                return thisOpacity;
            });
            node.style("fill-opacity", function(o) {
                thisOpacity = isConnected(d, o) ? 1 : opacity;
                return thisOpacity;
            });
            // also style link accordingly
            link.style("stroke-opacity", function(o) {
                return o.source === d || o.target === d ? 1 : opacity;
            });
            link.style("stroke", function(o){
                return o.source === d || o.target === d ? o.source.colour : "#ddd";
            });
        };
    }

    function mouseOut() {
        node.style("stroke-opacity", 1);
        node.style("fill-opacity", 1);
        link.style("stroke-opacity", 1);
        link.style("stroke", "#ddd");
    }

});
