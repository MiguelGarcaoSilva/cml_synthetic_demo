// window.cloropleth = Object.assing([], window.cloropleth, {
//     covid: {
//         StyleHandler: function(feature, context) {
//             const {
//                 classes,
//                 colorscale,
//                 style,
//                 colorProp
//             } = context.props.hideout;
//             const value = feature.properties[colorProp]; //get value to detemirne color
//             for (let i = 0; i < classes.length; ++i){
//                 if (value > classes[i]){
//                     style.fillColor = colorscale[i]; // set color based on color prop.
//                 }
//             }
//             return style;
//         }
//     }
// })

