<!--JAVASRRIPT-->
    <script type="text/javascript">
      var loadFile = function(event) {
      var image = document.getElementById('output');
      image.src = URL.createObjectURL(event.target.files[0]);
    };
    </script>
 <!--END OF JAVASRRIPT-->