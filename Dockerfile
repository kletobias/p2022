FROM ruby:3.1-alpine

# Install dependencies
RUN apk add --no-cache \
    build-base \
    gcc \
    cmake \
    git

# Set working directory
WORKDIR /app

# Copy Gemfile
COPY Gemfile* ./

# Install gems
RUN bundle install

# Copy the rest of the site
COPY . .

# Expose port
EXPOSE 4000

# Start Jekyll
CMD ["bundle", "exec", "jekyll", "serve", "--host", "0.0.0.0", "--livereload", "--force_polling"]