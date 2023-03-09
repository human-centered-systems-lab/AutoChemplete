import React from "react";
import {
  Card,
  CardContent,
  Typography,
  Box,
  Tooltip
} from "@material-ui/core";


const ModelOutputCard = (props) => {
  const { title } = props;
  return (
    <Card style={{ maxWidth: "40vw" }}>
      <CardContent>
        <Typography>Model prediction</Typography>
        <Box minHeight="0.5vh"></Box>
        <Tooltip arrow title={title}>
          <Typography noWrap gutterBottom variant="h5" component="h5">
            {title}
          </Typography>
        </Tooltip>
      </CardContent>
    </Card>
  );
};

export default ModelOutputCard;
