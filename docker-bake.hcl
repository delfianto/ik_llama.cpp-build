# Local build definitions for the two image variants.
#
# Prefer the justfile (`just cpu` / `just cuda` / `just all`) -- it updates the
# mirror, materialises .build/src, and passes the SHA and build number below.
# Invoking bake directly requires .build/src to already exist.

variable "REGISTRY" {
  default = "ghcr.io/delfianto"
}

variable "IMAGE" {
  default = "ik_llama.cpp"
}

# Commit the source tree in .build/src was extracted from. Recorded as an image
# label so `just check` can tell what each image was built from without a
# separate stamp file that could drift.
variable "IK_LLAMA_SHA" {
  default = "unknown"
}

variable "IK_LLAMA_BUILD_NUMBER" {
  default = "0"
}

variable "IK_LLAMA_PATCH_SHA" {
  default = "unknown"
}

group "default" {
  targets = ["cpu", "cuda"]
}

target "_common" {
  context    = "."
  dockerfile = "Dockerfile"
  # Upstream source, extracted from the local bare mirror by `just`.
  contexts = {
    ikllama = ".build/src"
  }
  # znver5 and the CUDA archs are both x86_64-only; never let bake try emulation.
  platforms = ["linux/amd64"]
  args = {
    IK_LLAMA_SHA          = IK_LLAMA_SHA
    IK_LLAMA_BUILD_NUMBER = IK_LLAMA_BUILD_NUMBER
  }
  labels = {
    "org.opencontainers.image.source"   = "https://github.com/ikawrakow/ik_llama.cpp"
    "org.opencontainers.image.title"    = "ik_llama.cpp"
    "org.opencontainers.image.revision" = IK_LLAMA_SHA
    "org.opencontainers.image.fastmtp-patch" = IK_LLAMA_PATCH_SHA
  }
}

target "cpu" {
  inherits = ["_common"]
  args = {
    VARIANT = "cpu"
  }
  tags = ["${REGISTRY}/${IMAGE}:cpu"]
  labels = {
    "org.opencontainers.image.description" = "ik_llama.cpp built for Zen 5, CPU only"
  }
}

target "cuda" {
  inherits = ["_common"]
  args = {
    VARIANT = "cuda"
  }
  tags = ["${REGISTRY}/${IMAGE}:cuda"]
  labels = {
    "org.opencontainers.image.description" = "ik_llama.cpp built for Zen 5 + NVIDIA Ampere/Ada/Hopper"
  }
}
