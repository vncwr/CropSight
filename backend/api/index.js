module.exports = (req, res) => {
  res.status(200).json({
    name: "CropSight Backend API",
    status: "online",
    endpoints: {
      summarize: "POST /api/summarize",
    },
    message: "CropSight LLM Proxy is active and ready."
  });
};
